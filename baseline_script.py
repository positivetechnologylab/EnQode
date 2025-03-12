# This is the script for the baseline technique used to evaluate EnQode. It generates 8 qubit circuits using Qiskit's
# amplitude embedding technique (https://journals.aps.org/pra/abstract/10.1103/PhysRevA.93.032318),
# and can be modified to generate amplitude embedding circuits tailored for the IBM Brisbane backend for MNIST,
# Fashion-MNIST, and CIFAR-10.

# Notes: this script downloads files in a folder TRAINING_DATA_FOLDER, and saves files in the folder DATA_FOLDER_NAME, and also saves the final CSV to that folder.
# This script also necessitates that a noise model pickle object exists in the PATH_TO_NOISE_MODEL variable,
# and also saves a file named "baseline_metrics_sim_{NUM_QUBITS}qubits.csv" containing the resulting metrics
# to DATA_FOLDER_NAME. It also saves intermediate results to DATA_FOLDER_NAME.

# Also, this code connects to the IBM Quantum Cloud to transpile to that exact backend. However, because no jobs are
# actually submitted to the IBM Cloud, this step is not strictly necessary; you can just compile to an IBM Backend that
# has the same properties as the real machine (e.g., Fake IBM Backends: https://docs.quantum.ibm.com/api/qiskit-ibm-runtime/fake-provider)
# For consistency to what was actually used to generate the results, I have included loading data from the cloud in the code.
# Note that this requires you to have a .env file at path ENV_FILE_PATH with an IBM Quantum token in the variable {IBMQ_TOKEN}.

# The data indices used for generating the results in the EnQode paper are provided by the CSV at path
# DATASET_INDICES_FILE.

# Import libraries for data processing
import numpy as np
from sklearn.decomposition import PCA

# Import library for file saving
import os

from utils import *

# Define hyperparameters for generated circuits
NUM_QUBITS = 8

DATA_FOLDER_NAME = "baseline_data"

DENSITY_MAT_FOLDER_NAME = "gen_density_mats_{NUM_QUBITS}qubits"

GEN_CIRCS_FOLDER_NAME = "gen_circs_{NUM_QUBITS}qubits"

RESULTS_FOLDER_NAME = "result_np_arrs"

DATASET_INDICES_FILE = "sampled_dataset_indices.csv"

TRAINING_DATA_FOLDER = "./training_data"

NOISE_MODEL_PATH = "noise_model_ibm_brisbane.pkl"

# Create directories for storing data (if that is desired)
os.makedirs(f"{DATA_FOLDER_NAME}/{DENSITY_MAT_FOLDER_NAME}", exist_ok=True)

os.makedirs(f"{DATA_FOLDER_NAME}/{GEN_CIRCS_FOLDER_NAME}", exist_ok=True)

os.makedirs(f"{DATA_FOLDER_NAME}/{RESULTS_FOLDER_NAME}", exist_ok=True)

# Define variables for IBM Quantum Cloud connection (used for obtaining an IBM backend; note this is not strictly necessary and you
# can replace this step with a FakeProvider)
ENV_FILE_PATH = ".env"

IBM_QUANTUM_CHANNEL = "ibm_quantum"

# TODO: Change this below to match the IBM instance you are using
IBM_INSTANCE = "hub/group/project"

def process_images(image_set, random_state=42):
    """
    Takes an input set of data, and performs PCA to reduce to 2 ** NUM_QUBITS so that the data
    can be embedded on NUM_QUBITS. In order to ensure that the resulting data is a valid quantum state,
    subtract the minimum value of each feature so that all features are positive, and then normalize
    each vector.

    :param image_set: a Numpy array representing the data to be amplitude embedded of shape (n_samples, data_dim). 
    :param random_state: an integer representing the random state for PCA.
    :return: a Numpy array representing each data vector as a valid quantum state that can be amplitude embedded on N_QUBITS
    of shape (n_samples, data_dim).
    """
    X_zero_normalized_array = image_set
    # Apply PCA to fit the data on NUM_QUBITS
    pca = PCA(n_components=2 ** NUM_QUBITS, random_state=random_state)
    X_zero_pca = pca.fit_transform(X_zero_normalized_array)
    # Make all PCA features positive
    # Compute the minimum value of each PCA component
    pca_mins = X_zero_pca.min(axis=0)

    # Shift PCA features to make all values positive
    X_zero_pca_shifted = X_zero_pca - pca_mins

    # Normalize each PCA vector to have unit norm
    # Compute the L2 norm of each PCA vector
    norms = np.linalg.norm(X_zero_pca_shifted, axis=1, keepdims=True)

    # Avoid division by zero
    norms[norms == 0] = 1e-8

    # Normalize each PCA vector to have unit norm
    # Reshape norms to (n_samples, 1) for broadcasting
    norms = norms.reshape(-1, 1)

    # Normalize the PCA vectors
    X_zero_pca_normalized = X_zero_pca_shifted / norms

    return X_zero_pca_normalized

# Import quantum circuit tools
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector

# Import simulation and statevector initialization tools
from qiskit_aer import AerSimulator

from qiskit import transpile

from qiskit.circuit.library import Initialize

import time

def count_single_two_qubit_gates(gate_counts):
    """
    Takes a dictionary that maps gate types to an integer representing the gate counts, and returns
    the single qubit and two qubit gate counts for a specified set of single and two qubit gates of consideration.

    :param gate counts: a dictionary mapping gate types to the number of gates
    :return: a tuple (single_qubit_count, two_qubit_count) representing the single and two qubit gate count in the specified
    valid single and two qubit gates
    """
    # Define sets of single-qubit and two-qubit gate names
    single_qubit_gates = {'id', 'rz', 'sx', 'x'}
    two_qubit_gates = {'ecr'}

    # Initialize counters
    single_qubit_count = 0
    two_qubit_count = 0

    # Count the gates
    for gate, count in gate_counts.items():
        if gate in single_qubit_gates:
            single_qubit_count += count
        elif gate in two_qubit_gates:
            two_qubit_count += count

    # print(f"Single-qubit gates: {single_qubit_count}")
    # print(f"Two-qubit gates: {two_qubit_count}")

    return single_qubit_count, two_qubit_count

def transpile_random_datapoint_baseline(normalized_vecs, backend, initial_layout, rand_data_index=None, optimization_level=0, seed=42):
    """
    Transpiles a specific input statevector onto the specified backend by first transpiling onto the U3 and CX universal gate set,
    and then transpiling that circuit onto the specified backend of interest. This is done to ensure that the statevector has
    an intermediate universal representation, irrespective of the backend of interest.

    :param normalized_vecs: a numpy array representing the dataset of normalized vectors to embed
    :param backend: an IBMBackend object representing the backend to transpile the statevector to
    :param initial_layout: a Layout object representing the initial layout for which to map the quantum states
    :param rand_data_index: an integer representing the index of the data in normalized_vecs to embed
    :param optimization_level: an integer representing the optimization level used in Qiskit's transpiler
    :param seed: an integer representing the random seed used for reproducibility; this can be used in transpilation

    :return: a tuple (circ_depth, total_gates, num_single_qubit, num_two_qubit, time_elapsed, transpiled_qc, transpiled_qc_backend) representing:
        - an integer that is the depth of the circuit
        - an integer that is the total number of gates in the circuit
        - an integer that is the number of single qubit gates in the circuit
        - an integer that is the number of two qubit gates in the circuit
        - a float representing the time elapsed for transpilation of the circuit
        - a QuantumCircuit object representing the intermediate circuit transpiled to U3 and CNOT
        - a QuantumCircuit object representing the final circuit transpiled to backend
    """

    if rand_data_index == None:
        print("execute_random_datapoint_baseline: rand_data_index is None")
        rand_data_index = np.random.randint(0, normalized_vecs.shape[0])

    normalized_rand_vec = normalized_vecs[rand_data_index] / np.linalg.norm(normalized_vecs[rand_data_index])
    # print(f"execute_random_datapoint_baseline, normalized_rand_vec: {normalized_rand_vec}")
    # print(f"execute_random_datapoint_baseline, np.linalg.norm(normalized_rand_vec): {np.linalg.norm(normalized_rand_vec)}")

    # Track the time it takes to embed the statevecctor on a quantum circuit.
    current_time = time.time()
    # Initialize the circuit with the normalized data
    # Ensure the amplitudes form a valid quantum state
    state = Statevector(normalized_rand_vec)
    # state.normalize()

    # Create an initialization object
    init_gate = Initialize(state, normalize=True)

    # Create a quantum circuit with the appropriate number of qubits
    num_qubits = state.num_qubits
    test_targ_statevec_prep = QuantumCircuit(num_qubits)

    # Apply the initialization to the qubits
    test_targ_statevec_prep.append(init_gate, test_targ_statevec_prep.qubits)

    # test_targ_statevec_prep.draw()
    # Transpile the circuit for the AerSimulator backend
    # print(f"execute_random_datapoint_baseline, np.linalg.norm(normalized_vecs[rand_data_index]): {np.linalg.norm(normalized_vecs[rand_data_index])}")
    # print(f"execute_random_datapoint_baseline, normalized_vecs[rand_data_index].shape: {normalized_vecs[rand_data_index].shape}")
    # print(f"execute_random_datapoint_baseline, test_targ_statevec_prep.num_qubits: {test_targ_statevec_prep.num_qubits}")

    
    transpiled_qc = transpile(test_targ_statevec_prep, basis_gates=['u3', 'cx'], optimization_level=optimization_level)
    
    transpiled_qc_backend = transpile(transpiled_qc, backend, optimization_level=optimization_level, initial_layout=initial_layout)

    # End the timer when the circuit has been transpiled to the specified backend.
    time_elapsed = time.time() - current_time

    gate_counts = transpiled_qc_backend.count_ops()

    # print(f'execute_random_datapoint_baseline, gate_counts: {gate_counts}')

    total_gates = sum(count for gate, count in gate_counts.items() if gate != 'measure')
    # print(f"Total number of gates (excluding measurements): {total_gates}")

    num_single_qubit, num_two_qubit = count_single_two_qubit_gates(gate_counts)

    circ_depth = transpiled_qc_backend.depth()

    return circ_depth, total_gates, num_single_qubit, num_two_qubit, time_elapsed, transpiled_qc, transpiled_qc_backend

# Perform Noisy Simulation

# For connecting to IBM Quantum Cloud and loading in credentials
from qiskit_ibm_runtime import QiskitRuntimeService

# Import IBM Quantum token
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=ENV_FILE_PATH)

token = os.getenv("IBMQ_TOKEN")

# Load IBM Quantum Account
QiskitRuntimeService.save_account(channel=IBM_QUANTUM_CHANNEL, token=token, instance=IBM_INSTANCE, overwrite=True, set_as_default=True)

service = QiskitRuntimeService(channel=IBM_QUANTUM_CHANNEL, instance=IBM_INSTANCE)

ibm_backend = service.backend("ibm_brisbane")

# Define a filter function to exclude 'rz' gates
def exclude_rz_gate(inst):
    """
    Checks if the instruction's name is 'rz'.

    :param inst: a QuantumCircuit instruction
    :return: a boolean indicating whether the name of the QuantumCircuit instruction is 'rz' or not.
    """
    return inst.operation.name != 'rz'

# For saving and loading quantum circuit objects
from qiskit import qpy

# Noisy Simulation Code

# For noisy simulation and fidelity calculation
from qiskit.quantum_info import DensityMatrix, state_fidelity

def perform_simulation(quantum_circ, noise_model=None):
    """
    Performs density matrix simulation on the given circuit with the specified noise model.

    :param quantum_circ: a QuantumCircuit that represents the circuit to be run
    :param noise_model: a NoiseModel object (or None) that represents the noise model to be applied
    :return: a DensityMatrix representing the resulting density matrix from the simulation.
    """
    # Initialize the simulator and add a density matrix intruction to the circuit.
    simulator = AerSimulator(method='density_matrix', noise_model=noise_model)
    # Note: the below line modifies quantum_circ
    quantum_circ.save_density_matrix()
    result = simulator.run(quantum_circ).result()
    density_matrix = result.data()['density_matrix']
    return density_matrix

def subset_circuit(orig_circuit, num_qubits=NUM_QUBITS):
    """
    Given a QuantumCircuit object, return the resulting subcircuit that operates only on the first num_qubits qubits.

    :param orig_circuit: a QuantumCircuit that represents the circuit to be subsetted
    :param num_qubits: an integer representing the number of qubits desired in the subcircuit
    :return: a QuantumCircuit that represents the subset circuit, that only operates on num_qubits qubits.
    """
    # Qubits and classical bits of interest
    qubits_of_interest = range(num_qubits)
    classical_bits_of_interest = range(num_qubits)

    # Initialize new circuit
    new_circuit = QuantumCircuit(num_qubits, num_qubits)

    # Create mappings
    qubit_mapping = {original_idx: new_idx for new_idx, original_idx in enumerate(qubits_of_interest)}
    classical_mapping = {original_idx: new_idx for new_idx, original_idx in enumerate(classical_bits_of_interest)}

    # Extract and add instructions
    for instr, qargs, cargs in orig_circuit.data:
        # Get indices of qubits involved in the instruction
        qubit_indices = [orig_circuit.qubits.index(qubit) for qubit in qargs]
        
        # Check if all qubits are among the qubits of interest
        if all(q in qubits_of_interest for q in qubit_indices):
            # Map the qubits to the new circuit's qubits
            new_qargs = [new_circuit.qubits[qubit_mapping[orig_circuit.qubits.index(q)]] for q in qargs]
            
            # Map classical bits if necessary
            if cargs:
                classical_indices = [orig_circuit.clbits.index(bit) for bit in cargs]
                new_cargs = [new_circuit.clbits[classical_mapping[idx]] for idx in classical_indices]
            else:
                new_cargs = []
            
            # Add the instruction to the new circuit
            new_circuit.append(instr, new_qargs, new_cargs)
    return new_circuit

baseline_metrics_keys = {"dataset", "class_label", "technique_type",
                         "comp_time_mean", "comp_time_stddev", "comp_times",
                         "simulated_fid_mean", "simulated_fid_stddev", "simulated_fids",
                         "noisy_fid_mean", "noisy_fid_stddev", "noisy_fids",
                         "sampled_data_idxs"}

import statistics

baseline_metrics_list = []

def get_reorder_qubits_circ(final_mapping: list, num_qubits=NUM_QUBITS) -> QuantumCircuit:
    """
    Returns a 'circuit' by inserting SWAP gates so that after all gates, based on 'final_mapping',
    qubit i occupies physical position i.
    
    :param final_mapping: A list of length n representing the final mapping p from some quantum circuit of interest,
        where final_mapping[i] = j means "qubit i ends at position j."
    :param num_qubits: An integer representing the total number of qubits in the circuit
    :return: A QuantumCircuit object with SWAP gates such that qubit i occupies physical position i.
    """

    swap_qc = QuantumCircuit(num_qubits)
    n = len(final_mapping)

    # 1) Compute the inverse permutation p_inv of final_mapping p
    #    p_inv[j] = i  iff  p[i] = j.
    p_inv = [0] * n
    for i in range(n):
        p_inv[final_mapping[i]] = i

    # 2) Decompose p_inv into cycles
    visited = [False] * n

    for start in range(n):
        if not visited[start]:
            cycle = []
            current = start
            # Collect one cycle by repeatedly applying p_inv
            while not visited[current]:
                cycle.append(current)
                visited[current] = True
                current = p_inv[current]
            
            # print(cycle)
            # 3) Implement this cycle (of length k) with k-1 swaps
            if len(cycle) > 1:
                for i in range(len(cycle) - 1):
                    # print(f'reorder_qubits, swapping qubit {cycle[i]} with {cycle[i+1]}')
                    swap_qc.swap(cycle[i], cycle[i+1])

    return swap_qc

# For applying a virtual SWAP permutation so that qubits are ordered correctly
from qiskit.quantum_info import Operator

# For modelling noise in simulation
from qiskit_aer.noise import NoiseModel

# For loading in and storing Python objects
import pickle

# Load in the noise model object
with open(NOISE_MODEL_PATH, 'rb') as file:
    noise_model = pickle.load(file)

# Standard library modules for comparison and output writing
import math
import sys

def add_baseline_metrics(dataset_name, class_label, backend, initial_layout, noise_model, res_list, subsampled_idxs=[], technique_type="baseline", backend_name="ibm_brisbane", seed=42):
    """
    Modifies res_list by adding relevant metrics for all images in dataset_name and class_label. Each image is amplitude embedded,
    transpiled to backend with layout initial_layout and run on noise_model for noisy simulation. The images to sample
    are specified by subsampled_idxs.
    
    :param dataset_name: A string representing the name of the dataset to amplitude embed data.
    :param class_label: A value representing the class for which images are to be sampled from.
    :param backend: An IBMBackend to transpile each amplitude embedded data onto.
    :param initial_layout: A value representing the layout on which virtual to physical qubits are to be initially mapped.
    :param noise_model: A NoiseModel representing the noise model to apply in noisy simulation.
    :param res_list: A list storing all of the relevant metrics for the images in the specified dataset and class.
    :param subsampled_idxs: A list of integers representing the indices in the input dataset to seelct for amplitude embedding.
    :param technique_type: A string representing the name of the technique, used when saving files.
    :param backend_name: A string representing the name of the backend used in transpilation, used when saving files.
    :param seed: An integer representing the seed (for reproducibility) used in the baseline technique.
    :return: Nothing; modifies res_list and saves files in DATA_FOLDER_NAME (assuming that folder and its subfolders exist; else,
    the program errs)
    """
    # Load the grayscale images for the specified dataset and class.
    img_set = get_images(dataset_name, class_label)

    # Perform PCA and normalize the images so that they are valid quantum statevectors.
    normalized_vecs = process_images(img_set)
    
    # Create a list that stores the compilation times, simulated fidelities, and noisy fidelities.
    metrics_accum = [[], [], []]

    # Perform amplitude embedding and compute compilation time, simulated, and noisy fidelities.
    for vec_idx in subsampled_idxs:
        # print(vec_idx)
        # We transpile with initial layout (0, 1, 2, ..., NUM_QUBITS) for consistency with EnQode.
        # Transpile the quantum circuit onto the backend of interest, using Qiskit's amplitude embedding technique.
        metrics_obj = transpile_random_datapoint_baseline(normalized_vecs, backend, initial_layout, rand_data_index=vec_idx, seed=seed)
        # Store the time it took to transpile this datapoint.
        metrics_accum[0].append(metrics_obj[-3])
        # Store the .qpy representation of the circuit.
        # with open(f'{DATA_FOLDER_NAME}/{GEN_CIRCS_FOLDER_NAME}/{dataset_name}_{class_label}_{vec_idx}_{backend_name}.qpy', 'wb') as file:
        #     qpy.dump(metrics_obj[-1], file)
        
        # Return a subset of the circuit that operates only on the first NUM_QUBITS (after transpiling to the IBM backend,
        # the resulting circuit may have more than 4 qubits, but the remaining qubits are unused).
        new_circuit = subset_circuit(metrics_obj[-1])
        # Copy the above circuit, as perform_simulation modifies new_circuit.
        new_circuit_copy = new_circuit.copy()
        # Compute the density matrix resulting from the circuit without noise and with a noise model.
        test_density_mat = perform_simulation(new_circuit)
        test_density_mat_noise = perform_simulation(new_circuit_copy, noise_model=noise_model)

        # Obtain the target density matrix, virtually permuted in such a way that it is in the same qubit ordering
        # as the resulting layout from transpilation. Note that the resulting layout from transpilation is not
        # necessarily the same as the initial layout.
        targ_density_mat = DensityMatrix(normalized_vecs[vec_idx])
        qc_layout = metrics_obj[-1].layout.final_index_layout()
        # Obtain the operations needed to evolve the target density matrix to be in the same permutation as the transpiled version
        swap_layout_circ = get_reorder_qubits_circ(qc_layout)
        swap_layout_op = Operator(swap_layout_circ)
        targ_density_mat_perm = targ_density_mat.evolve(swap_layout_op)

        # with open(f'{DATA_FOLDER_NAME}/{DENSITY_MAT_FOLDER_NAME}/{dataset_name}_{class_label}_{vec_idx}_{backend_name}_ideal_sim.pkl', 'wb') as file:
        #     pickle.dump(test_density_mat, file)
        # with open(f'{DATA_FOLDER_NAME}/{DENSITY_MAT_FOLDER_NAME}/{dataset_name}_{class_label}_{vec_idx}_{backend_name}_noisy_sim.pkl', 'wb') as file:
        #     pickle.dump(test_density_mat_noise, file)
        # with open(f'{DATA_FOLDER_NAME}/{DENSITY_MAT_FOLDER_NAME}/{dataset_name}_{class_label}_{vec_idx}_{backend_name}_target.pkl', 'wb') as file:
        #     pickle.dump(targ_density_mat, file)
        # with open(f'{DATA_FOLDER_NAME}/{DENSITY_MAT_FOLDER_NAME}/{dataset_name}_{class_label}_{vec_idx}_{backend_name}_target_perm.pkl', 'wb') as file:
        #     pickle.dump(targ_density_mat_perm, file)
        
        # Compute the noiseless and noisy fidelities.
        ideal_fid = state_fidelity(test_density_mat, targ_density_mat_perm)
        noisy_fid = state_fidelity(test_density_mat_noise, targ_density_mat_perm)
        # Log an error to stderr if the ideal fidelity is not sufficiently close to 1.
        if not math.isclose(ideal_fid, 1.0, rel_tol=1e-9):
            print(f"Ideal fidelity for {NUM_QUBITS} qubits, {dataset_name} dataset, {class_label} class label, {vec_idx} data point index, {backend_name} backend is not 1: {ideal_fid}",
                  file=sys.stderr)

        # Store the ideal and noisy fidelities.
        metrics_accum[1].append(ideal_fid)
        metrics_accum[2].append(noisy_fid)  
    
    # Store the compilation times, simulated fidelities, and noisy fidelities for the specified dataset and class.
    res_list.append({"dataset": dataset_name, "class_label": class_label, "technique_type": technique_type,
                     "comp_time_mean": statistics.mean(metrics_accum[0]), "comp_time_stddev": statistics.stdev(metrics_accum[0]), "comp_times": metrics_accum[0],
                     "simulated_fid_mean": statistics.mean(metrics_accum[1]), "simulated_fid_stddev": statistics.stdev(metrics_accum[1]), "simulated_fids": metrics_accum[1],
                     "noisy_fid_mean": statistics.mean(metrics_accum[2]), "noisy_fid_stddev": statistics.stdev(metrics_accum[2]), "noisy_fids": metrics_accum[2],
                     "sampled_data_idxs": subsampled_idxs})

    # Save the state of res_list into a numpy array
    np.save(f"{DATA_FOLDER_NAME}/{RESULTS_FOLDER_NAME}/baseline_metrics_{NUM_QUBITS}qubits_dataset_{dataset_name}_class_{class_label}_technique_{technique_type}_backend_{backend_name}.npy", np.array(res_list))

# Run the simulations
import pandas as pd
import ast

# Specify the datasets and classes from which to sample
test_datasets = ["CIFAR_10", "mnist_784", "Fashion-MNIST"]

test_classes = [1, 5, 6, 8, 9]

# Define a function to convert a string into a list
def parse_list_column(list_str):
    """
    Turns a string representation into a list; returns an empty list if the string is not a valid list.
    
    :param list: A string that is desired to be parsed into a list
    :return: A Python value that is the result of evaluating the string.
    """
    try:
        return ast.literal_eval(list_str)
    except (ValueError, SyntaxError):
        return []

# Load in the file containing the randomly generated indices to sample for each class and dataset (for reproducibility)
enqode_metrics_df = pd.read_csv(f'{DATASET_INDICES_FILE}', converters={'sampled_data_idxs': parse_list_column})

# enqode_metrics_df.head()

# Add data to a list to aggregate baseline metrics across datasets and classes
for dataset in test_datasets:
    for class_label in test_classes:
        # Find the data indices to subsample for the specifed dataset and class
        targ_row = enqode_metrics_df.loc[(enqode_metrics_df['dataset'] == dataset) & (enqode_metrics_df["class_label"] == class_label)]
        subsampled_idxs = targ_row['sampled_data_idxs'].values[0]
        # Compute the baseline metrics for the IBM Quantum backend and noise model, and a linear initial layout.
        # Aggregate all metrics into baseline_metrics_list.
        add_baseline_metrics(dataset, class_label, ibm_backend, range(NUM_QUBITS), noise_model, baseline_metrics_list, subsampled_idxs=subsampled_idxs)

# Convert the list to a Pandas dataframe and save it
baseline_runs_df = pd.DataFrame(baseline_metrics_list)
baseline_runs_df.to_csv(f"{DATA_FOLDER_NAME}/baseline_metrics_sim_{NUM_QUBITS}qubits.csv", index=False)
