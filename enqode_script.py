# This is the script for the EnQode technique in the paper. It generates 8 qubit circuits using a custom ansatz with
# fixed depth linear in the number of qubits, and can be modified to generate amplitude embedding circuits tailored for
# the IBM Brisbane backend for MNIST, Fashion-MNIST, and CIFAR-10.

# Notes: this script downloads files in a folder TRAINING_DATA_FOLDER, and saves files in the folder DATA_FOLDER_NAME, and also saves the final CSV to that folder.
# This script also necessitates that a noise model pickle object exists in the PATH_TO_NOISE_MODEL variable,
# and also saves a file named "enqode_metrics_{NUM_QUBITS}qubits.csv" containing the resulting metrics
# to DATA_FOLDER_NAME. It also saves intermediate results to DATA_FOLDER_NAME.

# Also, this code connects to the IBM Quantum Cloud to transpile to that exact backend. However, because no jobs are
# actually submitted to the IBM Cloud, this step is not strictly necessary; you can just compile to an IBM Backend that
# has the same properties as the real machine (e.g., Fake IBM Backends: https://docs.quantum.ibm.com/api/qiskit-ibm-runtime/fake-provider)
# For consistency to what was actually used to generate the results, I have included loading data from the cloud in the code.
# Note that this requires you to have a .env file at path ENV_FILE_PATH with an IBM Quantum token in the variable {IBMQ_TOKEN}.

# The data indices used for generating the results in the EnQode paper are provided by the CSV at path
# DATASET_INDICES_FILE.

# For fast numerical manipulation
import numpy as np
import matplotlib.pyplot as plt
# For dimensionality reduction and clustering
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from utils import *

# Define hyperparameters for generated circuits

# EnQode generates circuits that scale linearly with respect to the qubit count.
NUM_QUBITS = 8
NUM_LAYERS = NUM_QUBITS
PARAMS_PER_LAYER = NUM_LAYERS

num_params = NUM_LAYERS * PARAMS_PER_LAYER

DATA_FOLDER_NAME = "enqode_data"

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

import math

def determine_num_clusters(processed_image_set, distance_cutoff=0.95, random_state=42):
    """
    Finds the number of clusters in K-means such that the mean squared inner product (i.e., quantum fidelity)
    between any point and its centroid is at least distance_cutoff (set to be 0.95 in the paper).

    :param processed_image_set: a Numpy array representing the input dataset of shape (n_samples, data_dim).
    :param distance_cutoff: a float representing the threshold value to determine the number of clusters that is sufficient
    :param random_state: an integer used for reproducibility in K-means
    :return: an integer representing the number of clusters necessary to achieve at least distance_cutoff average fidelity.
    """
    X_zero_pca_normalized = processed_image_set
    # For speed, define the number of clusters that this function will explore
    cluster_counts = list(range(2, 5)) + list(range(5, 200, 5))  # Cluster counts from 5 to 50 in increments of 5

    # For each cluster value, compute the average fidelity between any point and its centroid. Terminate the
    # function when the average fidelity is above distance cutoff.
    for n_clusters in cluster_counts:
        # Perform k-means clustering and assign clusters.
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        labels = kmeans.fit_predict(X_zero_pca_normalized)
        # print("labels", labels)
        # print("labels.shape", labels.shape)
        
        # Compute the average inner product between each point and its centroid.
        total_innerprod = 0

        for (point_idx, img_vec) in enumerate(X_zero_pca_normalized):
            total_innerprod += np.dot(img_vec, kmeans.cluster_centers_[labels[point_idx]])
        
        total_innerprod /= X_zero_pca_normalized.shape[0]

        # print(f"Clusters: {n_clusters}, Average Distance: {total_innerprod}")

        # Return if the number of clusters is sufficient to satisfy the threshold.
        if (total_innerprod >= math.sqrt(distance_cutoff)):
            return n_clusters

    # Return the maximum sized cluster that was explored if no amount of clusters is sufficient to satisify the threshold.
    return cluster_counts[-1]

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

def perform_clustering(processed_image_set, n_clusters, random_state=42):
    """
    Performs k-means clustering on the specified dataset with the specified number of clusters.

    :param processed_image_set: a Numpy array representing the input dataset of shape (n_samples, data_dim).
    :param n_clusters: an integer representing the number of clusters to yield in K-means
    :param random_state: an integer used for reproducibility in K-means
    :return: (kmeans, get_cluster_info), where
        - kmeans is a KMeans object that is fit on the specified image dataset with the number of clusters
        - get_cluster_info is a function that takes in a data index and returns the assigned cluster label and centroid of that cluster
    """
    X_zero_pca_normalized = processed_image_set

    # Perform K-Means clustering on the normalized PCA features
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(X_zero_pca_normalized)
    labels = kmeans.labels_

    # Define a function to get cluster info
    def get_cluster_info(index):
        cluster_label = labels[index]
        cluster_centroid = kmeans.cluster_centers_[cluster_label]
        return cluster_label, cluster_centroid
    
    return kmeans, get_cluster_info

# Import quantum circuit manipulation tools
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.circuit import ParameterVector

# Import symbolic quantum circuit tools
from qiskit_symb.quantum_info import Operator as SymOperator
import sympy as sym

def generate_Xshape_arrow_circ(params, layers, init=False, post=False, params_per_layer=PARAMS_PER_LAYER, non_phase=False):
    """
    Creates a hybrid cross and arrow shape circuit as described in the EnQode paper, parameterized by params with layers
    layers. May also pre-rotate and post-rotate the circuit accordingly for easier optimization in the 'phase plane' (as
    described in the paper).

    :param params: a Numpy array representing the parameters of the quantum circuit. The expected dimension of params is (at least)
    params_per_layer * layers.
    :param layers: an integer representing the number of layers in the quantum circuit.
    :param init: a Boolean indicating whether or not each qubit should be initially rotated to a superposition state.
    :param post: a Boolean indicating whether or not each qubit should be rotated from the 'x-y' plane to the 'x-z' plane of
    each qubit's Bloch sphere.
    :param params_per_layer: an integer representing the number of parameters in each layer of the quantum circuit.
    :param non_phase: a Boolean indicating whether or not Ry and CX gates should be used instead of Rz and CY Gates.
    :return: a QuantumCircuit parameterized by params.
    """

    # Initialize circuit
    circuit = QuantumCircuit(NUM_QUBITS)

    # If init, initialize each qubit to the |+> state for non_phase or the |i> state.
    if init == True:
        for qubit_num in range(NUM_QUBITS):
            if (non_phase):
                circuit.ry(np.pi/2, qubit_num)
            else:
                circuit.rx(-np.pi/2, qubit_num)


    # In an alternating fashion, add an X-shape or arrow shape two qubit gate structure in each layer.
    for layer in range(layers):
        # For even layers, add an X-shape two qubit gate structure.
        if (layer % 2 == 0):
            # Find the starting parameter indices for this layer.
            base_idx = layer * params_per_layer
            # Add a parameterized Ry gate in the x-z plane or Rz gate in the x-y plane for each qubit.
            for qubit_num in range(NUM_QUBITS):
                param_idx = base_idx + qubit_num * int((params_per_layer / NUM_QUBITS))
                if (non_phase):
                    circuit.ry(params[param_idx], qubit_num)
                else:
                    circuit.rz(params[param_idx], qubit_num)
            
            # Add two qubit gates in an alternating X-like structure.
            for qubit_num in range(0, NUM_QUBITS - 1, 2):
                if (non_phase):
                    circuit.cx(qubit_num, qubit_num + 1)
                else:
                    circuit.cy(qubit_num, qubit_num + 1)
            
            for qubit_num in range(1, NUM_QUBITS - 1, 2):
                if (non_phase):
                    circuit.cx(qubit_num, qubit_num + 1)
                else:
                    circuit.cy(qubit_num, qubit_num + 1)
            
            for qubit_num in range(0, NUM_QUBITS - 1, 2):
                if (non_phase):
                    circuit.cx(qubit_num, qubit_num + 1)
                else:
                    circuit.cy(qubit_num, qubit_num + 1)
        else:
            # Find the starting parameter indices for this layer.
            base_idx = layer * params_per_layer
            # Add a parameterized Ry gate in the x-z plane or Rz gate in the x-y plane for each qubit.
            for qubit_num in range(NUM_QUBITS):
                param_idx = base_idx + qubit_num * int((params_per_layer / NUM_QUBITS))
                if (non_phase):
                    circuit.ry(params[param_idx], qubit_num)
                else:
                    circuit.rz(params[param_idx], qubit_num)

            # Add two qubit gates in an alternating arrow-like structure.
            for qubit_num in range(0, NUM_QUBITS - 1, 2):
                if (non_phase):
                    circuit.cx(qubit_num, qubit_num + 1)
                else:
                    circuit.cy(qubit_num, qubit_num + 1)
            
            for qubit_num in range(1, NUM_QUBITS - 1, 2):
                if (non_phase):
                    circuit.cx(qubit_num, qubit_num + 1)
                else:
                    circuit.cy(qubit_num, qubit_num + 1)

    # If post is True, then rotate each qubit from the x-y plane to the x-z plane.
    if post == True:
        for qubit_num in range(NUM_QUBITS):
            circuit.rx(-np.pi/2, qubit_num)
            circuit.ry(-np.pi/2, qubit_num)

    return circuit

# For randomly sampling data
import random

def get_nl_equations(qc, param_vec):
    """
    Create a single lambdified function that outputs the complete complex statevector.

    :param qc: A QuantumCircuit parameterized by symbolic parameters.
    :param param_vec: A ParameterVector that is used to parameterize qc.
    :return: a tuple (statevector_func, psi_expr, sym_params), where
      - statevector_func is a function that, given parameters, returns the complete statevector (as a NumPy array).
      - psi_expr is a list of symbolic expressions for each entry.
      - sym_params is a tuple of the "clean" sympy parameter symbols.
    """
    # Convert the quantum circuit to a Symbolic operator, and extract the first column from the unitary
    # as we are only concerned with how the unitary evolves |0> -> U|0>.
    opY = SymOperator(qc)
    symY_u = opY.to_sympy()
    column = sym.simplify(symY_u[:, 0])
    
    # Create a tuple of clean sympy symbols for the parameters.
    sym_params = sym.symbols(f"x:{len(param_vec)}", real=True)
    # Create a tuple for the original parameters to be replaced.
    t = sym.symbols([f"t[{i}]" for i in range(len(param_vec))])
    param_swap_list = dict(zip(t, sym_params))
    
    psi_expr = []  # symbolic expressions for each entry
    for entry in column:
        # Replace t[{i}] with x:{i} as the parameters, and simplify each value in the column
        entry = sym.nsimplify(entry.subs(param_swap_list))
        psi_expr.append(entry)
    
    # Combine the list of expressions into a single sympy Matrix.
    psi_vec_expr = sym.Matrix(psi_expr)
    
    # Lambdify the complete statevector.
    statevector_func = sym.lambdify((*sym_params,), psi_vec_expr, modules='numpy')
    return statevector_func, psi_expr, sym_params

def compute_gradient_func(psi_expr, sym_params):
    """
    Given a list of symbolic expressions psi_expr (the statevector entries)
    and a tuple of sympy symbols sym_params, returns a gradient function
    with signature grad_func(x0, x1, ..., x_{n-1}, T0, T1, ..., T_{N-1}),
    where N = len(psi_expr) and n = len(sym_params). The returned function
    computes the gradient of the negative fidelity objective:
    
       f(x) = -|F(x)|^2, where F(x) = Σ_i T_i * conj(ψ_i(x))
    
    The gradient is given by:
    
       ∇f(x) = -2 * Re{ conj(F(x)) * (dF/dx) }
       
    where dF/dxₖ = Σ_i T_i * conj(∂ψ_i(x)/∂xₖ).
    
    This implementation is optimized by precompiling the statevector and its Jacobian
    as NumPy-callable functions.

    :param psi_expr: a list of symbolic values for each entry of the statevector
    :param sym_params: a tuple of the parameter symbols to use for symbolic evaluation of the statevector
    :return: a function that takes in the datapoint of interest followed by the parameters in the circuit and
    returns the gradient
    """
    # Lambdify the statevector (psi_expr) as a function of sym_params.
    psi_func = sym.lambdify(sym_params, psi_expr, modules='numpy')
    
    # Compute the Jacobian matrix of psi_expr with respect to sym_params.
    J_expr = sym.Matrix(psi_expr).jacobian(sym_params)
    jacobian_func = sym.lambdify(sym_params, J_expr, modules='numpy')
    
    def grad_func(*args):
        """
        Expects:
        :param *args: a tuple with the first n entries being the parameter values (x0, x1, ..., x_{n-1})
        and the following N entries being the target amplitudes (T0, T1, ..., T_{N-1}).
        :return: a NumPy array of length n containing the gradient of the negative fidelity.
        """
        n_params = len(sym_params)
        # Extract parameter vector x and target vector T.
        x = np.array(args[:n_params])
        T = np.array(args[n_params:])
        
        # Evaluate the statevector at x (as a flat array).
        psi = np.array(psi_func(*x)).flatten()  # shape: (N,)
        # Compute F(x) = Σ_i T_i * conj(ψ_i(x))
        F = np.dot(np.conjugate(psi), T)
        
        # Evaluate the Jacobian at x. J is an (N x n_params) array.
        J = np.array(jacobian_func(*x))
        # Compute dF/dx = Σ_i T_i * conj(J[i, :]) via dot–product.
        dF = np.dot(np.conjugate(J).T, T)
        
        # Compute the gradient: -2 * Re{ conj(F) * dF }.
        grad = -2 * np.real(np.conjugate(F) * dF)
        return grad
    
    return grad_func

# Import fidelity and minimization function for variational optimization
from qiskit.quantum_info import state_fidelity
from scipy.optimize import minimize

def solve_nl_equations_fid(statevector_func, grad_func, targ_statevec, num_params, x0=None):
    """
    Optimize the negative fidelity metric using a single statevector function.
    
    :param statevector_func: a lambdified function that returns the full statevector (complex) for given parameters.
    :param grad_func: a precomputed gradient function that takes in the datapoint and parameters.
    :param targ_statevec: a complex numpy array representing the target statevector.
    :param num_params: an integer representing the number of parameters in the circuit.
    :param x0: (optional) initial guess for the parameters.
    :return res: the optimization result from scipy.optimize.minimize.
    """

    def fidelity_metric(x_values):
        """
        Optimize the negative fidelity metric using a single statevector function.
        
        :param x_values: a numpy array containing the parameters of interest
        :return: a float representing the negative fidelity to the target statevector 
        """
        # Directly evaluate the complete statevector.
        cur_statevec = np.array(statevector_func(*x_values)).flatten()
        inn_prod = np.vdot(cur_statevec, targ_statevec)
        return -np.abs(inn_prod)**2

    # Initialize all parameters as 0 for an initial guess.
    if x0 is None:
        x0 = np.zeros(num_params)

    # Employ L-BFGS-B gradient based optimization with a symbolic gradient.
    res = minimize(
        fidelity_metric,
        x0,
        method='L-BFGS-B',
        # bounds=bounds,
        jac=lambda x: np.real(np.array(grad_func(*x, *targ_statevec), dtype=float))
    )
    return res

# Import simulation tools
from qiskit_aer import AerSimulator

from qiskit.quantum_info import DensityMatrix

from qiskit import transpile

# Import tools for connecting to IBM Quantum Cloud
from qiskit_ibm_runtime import QiskitRuntimeService

from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=ENV_FILE_PATH)

token = os.getenv("IBMQ_TOKEN")

QiskitRuntimeService.save_account(channel=IBM_QUANTUM_CHANNEL, token=token, instance=IBM_INSTANCE, overwrite=True, set_as_default=True)

# To access saved credentials for the IBM quantum channel
service = QiskitRuntimeService(channel=IBM_QUANTUM_CHANNEL, instance=IBM_INSTANCE)

def get_inv_post_proc_op():
    """
    Return a quantum circuit that rotates data from the x-y plane to the x-z plane for each qubit.

    :return: a QuantumCircuit that has single qubit rotations from the x-y plane to the x-z plane for each qubit.
    """
    post_proc = QuantumCircuit(NUM_QUBITS)
    post_proc.rx(-np.pi/2, range(NUM_QUBITS))
    post_proc.ry(-np.pi/2, range(NUM_QUBITS))
    inv_post_proc = post_proc.inverse()
    return Operator(inv_post_proc).data

def basis_change_input_vec(targ_vec, inv_post_proc_op):
    """
    Given a target statevector, rotate the data such that it is from the x-z plane to the x-y plane.

    :return: a statevector that is rotated to be in the x-y plane via single-qubit rotations.
    """
    new_test_targ_weights = inv_post_proc_op @ targ_vec
    return new_test_targ_weights

def get_optimization_components(circ_name, circ_gen_func):
    """
    Obtains the symbolic statevector and gradient functions, as well as symbolic expressions used for
    symbolic optimization.

    :param circ_name: a string representing the name of the circuit to use in optimization
    :param circ_gen_func: a function that takes in parameters and a number of layers, and outputs a parameterized quantum circuit
    :return: a tuple (statevector_func, psi_expr, sym_params, grad_func, circ_params) where:
        - statevector_func outputs a statevector given input parameters
        - psi_expr contains symbolic values for each entry in the statevector
        - sym_params is an array of symbolic parameters used in statevector_func
        - grad_func is a function that computes the gradient given a target statevector and the parameters of interest for the symbolic statevector
        - circ_params is the symbolic parameters used in the quantum circuit
    """
    # Create symbolic parameters for the quantum circuit
    circ_params = ParameterVector("t", length=PARAMS_PER_LAYER * NUM_LAYERS)
    if circ_name == "layer_with_block":
        circ_params = ParameterVector("t", length=(PARAMS_PER_LAYER + int(PARAMS_PER_LAYER / 2)) * NUM_LAYERS)
    # Parameterize the circuit with the symbolic parameters
    gen_circ = circ_gen_func(circ_params, NUM_LAYERS, init=True)
    # Get the non-linear equations (statevector symbolic functions)
    statevector_func, psi_expr, sym_params = get_nl_equations(gen_circ, circ_params)
    # Precompute the gradient function once given the symbolic statevector
    grad_func = compute_gradient_func(psi_expr, sym_params)
    return statevector_func, psi_expr, sym_params, grad_func, circ_params

# For timing the duration of offline and online optimization
import time

def perform_datapoint_optimization(centroid, statevector_func,
                                   psi_expr, sym_params, grad_func, inv_post_proc_op, num_params, x0=None):
    """
    For a given datapoint (centroid), performs optimization of the parameters in statevector_func to obtain a close
    estimate that minimizes fidelity.

    :param centroid: a Numpy array that represents the target datapoint for which fidelity is minimized towards
    :param statevector_func: a function that takes in parameters and outputs the statevector for those parameters
    :param psi_expr: a list containing symbolic values for each entry in the statevector (unused)
    :param sym_params: an array of symbolic parameters used in statevector_func (unused)
    :param grad_func: a function that computes the gradient given a target statevector and the parameters of interest for the symbolic statevector
    :param inv_post_proc_op: a QuantumCircuit that is applied to rotate the input data such that it is easy to optimize
    :param num_params: an integer representing the number of parameters for the symbolic statevector
    :param x0: a Numpy array representing the initial guess of parameters used for optimizing the symbolic statevector to the target statevector
    :return: an OptimizeResult containing the results of the optimization of parameters
    """
    # Perform a rotation on the target data to be in the x-y plane for optimization
    norm_centroid_basischange = basis_change_input_vec(centroid, inv_post_proc_op)
    # Perform optimization given symbolic functions to compute the statevector and gradient for the target data
    res_centroid = solve_nl_equations_fid(
        statevector_func,
        grad_func,
        norm_centroid_basischange,
        num_params, x0=x0
    )
    return res_centroid

def perform_offline_optimization(normalized_vecs, num_clusters,
                                 statevector_func,
                                 psi_expr, sym_params, grad_func, inv_post_proc_op, num_params):
    """
    Offline optimization: cluster the data and optimize for each cluster centroid.

    :param normalized_vecs: a Numpy array of dimension (n_samples, data_dim) containing normalized statevectors
    :param num_clusters: an integer representing the number of clusters to use when clustering the data
    :param statevector_func: a function that takes in parameters and outputs the statevector for those parameters
    :param psi_expr: a list containing symbolic values for each entry in the statevector
    :param sym_params: an array of symbolic parameters used in statevector_func
    :param grad_func: a function that computes the gradient given a target statevector and the parameters of interest for the symbolic statevector
    :param inv_post_proc_op: a QuantumCircuit that is applied to rotate the input data such that it is easy to optimize
    :param num_params: an integer representing the number of parameters for the symbolic statevector
    :return: a tuple (centroid_idx_to_res, get_cluster_info), where:
        - centroid_idx_to_res maps the centroid index to the optimization result for that centroid index
        - get_cluster_info takes in the index of the datapoint of interest and returns the cluster that the datapoint is assigned to
        as well as the centroid of that cluster
    """
    # Perform clustering on the input dataset
    kmeans, get_cluster_info = perform_clustering(normalized_vecs, num_clusters)
    # Perform optimization to map each centroid of each cluster to the optimization result
    centroid_idx_to_res = {}
    for centroid_idx, centroid in enumerate(kmeans.cluster_centers_):
        res_centroid = perform_datapoint_optimization(
            centroid, statevector_func,
            psi_expr, sym_params, grad_func, inv_post_proc_op, num_params
        )
        centroid_idx_to_res[centroid_idx] = res_centroid
    # Return the centroid optimization results, as well as a mapping from data indices to centroids and clusters
    return centroid_idx_to_res, get_cluster_info

# Perform a one-time call to generate the components for optimization.
# Note that these components can be generated once, stored as variables, and imported in to start your optimization for any dataset of interest
# (these values are completely independent of any data you are embedding for).
statevector_func, psi_expr, sym_params, grad_func, circ_params = get_optimization_components("xshape_arrow_hybrid", generate_Xshape_arrow_circ)

# Perform Online Optimization
def perform_online_optimization(normalized_vecs, get_cluster_info, centroid_label_to_res,
                                statevector_func,
                                psi_expr, sym_params, grad_func,
                                inv_post_proc_op, param_vec, final_symb_circ, rand_data_index=None, seed=42):
    """
    Online optimization: for a randomly chosen data point, start from the corresponding cluster solution
    and optimize further.

    :param normalized_vecs: a Numpy array of dimension (n_samples, data_dim) containing normalized statevectors
    :param get_cluster_info: a function takes in the index of the datapoint of interest and returns the cluster that the datapoint is assigned to
        as well as the centroid of that cluster
    :param centroid_label_to_res: an object that maps the centroid index to the optimization result for that centroid index
    :param statevector_func: a function that takes in parameters and outputs the statevector for those parameters
    :param psi_expr: a list containing symbolic values for each entry in the statevector
    :param sym_params: an array of symbolic parameters used in statevector_func
    :param grad_func: a function that computes the gradient given a target statevector and the parameters of interest for the symbolic statevector
    :param inv_post_proc_op: a QuantumCircuit that is applied to rotate the input data such that it is easy to optimize
    :param param_vec: A ParameterVector that is used to parameterize the quantum circuit of interest.
    :param final_symb_circ: A QuantumCircuit parameterized by param_vec, allowing for efficient parameterization of the circuit with specified parameters.
    :return: a QuantumCircuit that is parameterized by the result of the optimization.
    """
    # Select a random data index if a data index is not specified.
    if rand_data_index is None:
        rand_data_index = np.random.randint(0, normalized_vecs.shape[0])
    
    # Find the parameters used for optimizing the centroid, and use those as a warm start for optimizing
    # the datapoint of interest.
    cluster_label, cluster_centroid = get_cluster_info(rand_data_index)
    centroid_res = centroid_label_to_res[cluster_label]
    # Perform optimization towards the datapoint of interest.
    res_datapoint = perform_datapoint_optimization(
        normalized_vecs[rand_data_index],
        statevector_func,
        psi_expr, sym_params, grad_func,
        inv_post_proc_op, len(param_vec),
        x0=centroid_res.x
    )
    # Update the symbolic circuit with the optimized parameters.
    param_dict = {param_vec[i]: res_datapoint.x[i] for i in range(len(param_vec))}
    parameterized_circ = final_symb_circ.assign_parameters(param_dict)
    return parameterized_circ

# Define the IBM Backend to transpile to.
ibm_backend = service.backend("ibm_brisbane")

# Create the symbolic circuit with the specified number of layers and transpile it to the target hardware.
# Note that this step is also completely independent of the data that you would like to embed, and thus can be performed once for all data
# to embed; the circuit object can be stored and loaded.
final_symb_circ = generate_Xshape_arrow_circ(circ_params, NUM_LAYERS, init=True, post=True)
transpiled_symb_circ = transpile(final_symb_circ, ibm_backend, optimization_level=0, initial_layout=range(NUM_QUBITS), seed_transpiler=42)

def perform_simulation(quantum_circ, noise_model=None, copy=False, seed=42):
    """
    Performs density matrix simulation on the given circuit with the specified noise model.

    :param quantum_circ: a QuantumCircuit that represents the circuit to be run
    :param noise_model: a NoiseModel object (or None) that represents the noise model to be applied
    :param copy: a Boolean indicating whether or not the circuit should be copied before the simulation is ran
    :param seed: an integer representing the random seed (unused)
    :return: a DensityMatrix representing the resulting density matrix from the simulation.
    """
    # Initialize the simulator and add a density matrix intruction to the circuit.
    if copy:
        quantum_circ = quantum_circ.copy()
    simulator = AerSimulator(method='density_matrix', noise_model=noise_model)
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

# For noisy simulation
from qiskit_aer.noise import NoiseModel

# Load in the noise model object
import pickle
with open(NOISE_MODEL_PATH, 'rb') as file:
    noise_model = pickle.load(file)

# For metrics aggregation
import statistics

# Columns in resulting dataframe
enqode_metrics_keys = {"dataset", "class_label", "technique_type", "offline_comp_time",
                         "online_comp_time_mean" "online_comp_time_stddev", "simulated_fid_mean", "simulated_fid_stddev",
                         "noisy_fid_mean", "noisy_fid_stddev", "sampled_data_idxs"}

# Aggregates metrics
enqode_metrics_list = []

# For saving circuits
from qiskit import qpy

def add_enqode_metrics(dataset_name, class_label, statevector_func, psi_expr, sym_params, grad_func, inv_post_proc_op, param_vec, final_symb_circ, noise_model, res_list, num_sampled_points=500, technique_type="EnQode", backend_name="ibm_brisbane", subsampled_idxs=None):
    """
    Modifies res_list by adding relevant metrics for all images in dataset_name and class_label. Each image is variationally optimized to find an approximate
    close image, and a resulting circiuit transpiled to backend with layout initial_layout and run on noise_model for noisy simulation is generated. The images to sample
    are specified by subsampled_idxs.
    
    :param dataset_name: A string representing the name of the dataset to amplitude embed data.
    :param class_label: A value representing the class for which images are to be sampled from.
    :param statevector_func: a function that takes in parameters and outputs the statevector for those parameters
    :param psi_expr: a list containing symbolic values for each entry in the statevector
    :param sym_params: an array of symbolic parameters used in statevector_func
    :param grad_func: a function that computes the gradient given a target statevector and the parameters of interest for the symbolic statevector
    :param inv_post_proc_op: a QuantumCircuit that is applied to rotate the input data such that it is easy to optimize
    :param param_vec: A ParameterVector that is used to parameterize the quantum circuit of interest.
    :param final_symb_circ: A QuantumCircuit parameterized by param_vec, allowing for efficient parameterization of the circuit with specified parameters.
    :param noise_model: A NoiseModel representing the noise model to apply in noisy simulation.
    :param res_list: A list storing all of the relevant metrics for the images in the specified dataset and class.
    :param num_sampled_points: An integer representing the number of points to be sampled from the dataset for approximate amplitude encoding.
    :param technique_type: A string representing the name of the technique, used when saving files.
    :param backend_name: A string representing the name of the backend used in transpilation, used when saving files.
    :param subsampled_idxs: A list of integers representing the indices in the input dataset to seelct for amplitude embedding.

    :return: Nothing; modifies res_list and saves files in DATA_FOLDER_NAME (assuming that folder and its subfolders exist; else,
    the program errs)
    """
    # Load the images for the specified dataset and normalize them.
    img_set = get_images(dataset_name, class_label)

    normalized_vecs = process_images(img_set)

    # Find the number of clusters based on the data
    num_clusters = determine_num_clusters(normalized_vecs) 

    # Perform optimization of each cluster centroid and store that dta.

    cur_time = time.time()

    centroid_label_to_res, offline_get_cluster_info = perform_offline_optimization(normalized_vecs, num_clusters, statevector_func,
                                                                                   psi_expr, sym_params, grad_func, inv_post_proc_op, len(param_vec))

    offline_time_elapsed = time.time() - cur_time

    # Create a list that stores the compilation times, simulated fidelities, and noisy fidelities.
    metrics_accum = [[], [], []]

    # Ensure the list has at least num_sampled_points elements
    if len(normalized_vecs) < num_sampled_points:
        raise ValueError(f"The list must contain at least {num_sampled_points} elements.")

    # Use the specified indices for sampling data; otherwise, randomly sample indices from the dataset
    if subsampled_idxs != None:
        sampled_indices = subsampled_idxs
    
    else:
        # Generate a list of indices corresponding to the data points
        indices = list(range(len(normalized_vecs)))

        # Randomly sample 500 unique indices
        sampled_indices = random.sample(indices, num_sampled_points)

    # Perform approximate amplitude encoding for each datapoint by finding optimizing for parameters.
    for vec_idx in sampled_indices:
        # print(vec_idx)
        cur_online_time = time.time()
        param_circ = perform_online_optimization(normalized_vecs, offline_get_cluster_info, centroid_label_to_res,
                                                 statevector_func,
                                                  psi_expr, sym_params, grad_func,
                                                    inv_post_proc_op, param_vec, final_symb_circ, rand_data_index=vec_idx)
        online_time_elapsed = time.time() - cur_online_time
        metrics_accum[0].append(online_time_elapsed)

        # Store the .qpy representation of the circuit.
        # with open(f'enqode_data/gen_circs_{NUM_QUBITS}qubits_symbgrad_gradopt/{dataset_name}_{class_label}_{vec_idx}_{backend_name}.qpy', 'wb') as file:
        #     qpy.dump(param_circ, file)

        # Return a subset of the circuit that operates only on the first NUM_QUBITS (after transpiling to the IBM backend,
        # the resulting circuit may have more than 4 qubits, but the remaining qubits are unused).
        new_circuit = subset_circuit(param_circ)
        # Copy the above circuit, as perform_simulation modifies new_circuit.
        new_circuit_copy = new_circuit.copy()
        # Compute the density matrix resulting from the circuit without noise and with a noise model.
        test_density_mat = perform_simulation(new_circuit)
        test_density_mat_noise = perform_simulation(new_circuit_copy, noise_model=noise_model)
        targ_density_mat = DensityMatrix(normalized_vecs[vec_idx])

        # Save the generated density matrices, if desired.
        # with open(f'enqode_data/gen_density_mats_{NUM_QUBITS}qubits_symbgrad_gradopt/{dataset_name}_{class_label}_{vec_idx}_{backend_name}_ideal_sim.pkl', 'wb') as file:
        #     pickle.dump(test_density_mat, file)
        # with open(f'enqode_data/gen_density_mats_{NUM_QUBITS}qubits_symbgrad_gradopt/{dataset_name}_{class_label}_{vec_idx}_{backend_name}_noisy_sim.pkl', 'wb') as file:
        #     pickle.dump(test_density_mat_noise, file)
        # with open(f'enqode_data/gen_density_mats_{NUM_QUBITS}qubits_symbgrad_gradopt/{dataset_name}_{class_label}_{vec_idx}_{backend_name}_target.pkl', 'wb') as file:
        #     pickle.dump(targ_density_mat, file)

        # Compute the noiseless and noisy fidelities.
        ideal_fid = state_fidelity(test_density_mat, targ_density_mat)
        noisy_fid = state_fidelity(test_density_mat_noise, targ_density_mat)

        # Store the ideal and noisy fidelities.
        metrics_accum[1].append(ideal_fid)
        metrics_accum[2].append(noisy_fid)        
    
    res_list.append({"dataset": dataset_name, "class_label": class_label, "technique_type": technique_type,
                     "offline_comp_time": offline_time_elapsed,
                     "online_comp_time_mean": statistics.mean(metrics_accum[0]), "online_comp_time_stddev": statistics.stdev(metrics_accum[0]), "online_comp_times": metrics_accum[0],
                     "simulated_fid_mean": statistics.mean(metrics_accum[1]), "simulated_fid_stddev": statistics.stdev(metrics_accum[1]), "simulated_fids": metrics_accum[1],
                     "noisy_fid_mean": statistics.mean(metrics_accum[2]), "noisy_fid_stddev": statistics.stdev(metrics_accum[2]), "noisy_fids": metrics_accum[2],
                     "sampled_data_idxs": sampled_indices})

# Specify the datasets and classes from which to sample
test_datasets = ["CIFAR_10", "mnist_784", "Fashion-MNIST"]

test_classes = [1, 5, 6, 8, 9]

# For parsing list data
import ast

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

# For storing data in a dataframe
import pandas as pd

# Load in the file containing the randomly generated indices to sample for each class and dataset (for reproducibility)
enqode_metrics_df = pd.read_csv(f'{DATASET_INDICES_FILE}', converters={'sampled_data_idxs': parse_list_column})

# Get the operation used to rotate the input data so that it is favorable for optimization
inv_post_proc_op = get_inv_post_proc_op()

# Add data to a list to aggregate baseline metrics across datasets and classes
for dataset in test_datasets:
    for class_label in test_classes:
        # Find the data indices to subsample for the specifed dataset and class
        targ_row = enqode_metrics_df.loc[(enqode_metrics_df['dataset'] == dataset) & (enqode_metrics_df["class_label"] == class_label)]
        subsampled_idxs = targ_row['sampled_data_idxs'].values[0]
        # Compute the baseline metrics for the IBM Quantum backend and noise model, and a linear initial layout.
        # Aggregate all metrics into baseline_metrics_list.
        add_enqode_metrics(dataset, class_label, statevector_func, psi_expr, sym_params, grad_func, inv_post_proc_op, circ_params, transpiled_symb_circ, noise_model, enqode_metrics_list, subsampled_idxs=subsampled_idxs)

# Convert the list to a Pandas dataframe and save it
enqode_runs_df = pd.DataFrame(enqode_metrics_list)
enqode_runs_df.to_csv(f"{DATA_FOLDER_NAME}/enqode_metrics_{NUM_QUBITS}qubits.csv", index=False)
