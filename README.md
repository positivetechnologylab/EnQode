# EnQode: Fast Amplitude Embedding for Quantum Machine Learning Using Classical Data

## Overview
EnQode is a novel amplitude encoding framework that employs symbolic optimization with warm-start centroid learning.

## Usage
1. In your terminal, run `git clone https://github.com/positivetechnologylab/enqode.git` to clone the repository.
2. Run `cd enqode`.
3. Create a virtual environment if necessary (our code uses Python 3.12.3), and run `pip install -r requirements.txt` to install the requirements.
4. Run either `baseline_script.py` or `enqode_script.py` to run the Baseline or EnQode technique on MNIST, Fashion-MNIST, and CIFAR-10 datasets.
    - The variable `NUM_QUBITS` can be changed to indicate the number of qubits to perform embedding.
5. The results will be stored in either `baseline_data/baseline_metrics_sim_{NUM_QUBITS}qubits.csv` or `enqode_data/enqode_metrics_{NUM_QUBITS}qubits.csv`, for the baseline and EnQode technique, respectively.

## Requirements
The requirements and specific versions are provided in `requirements.txt`.
For loading in an IBM Quantum account, the script finds a .env file located at `ENV_FILE_PATH` and finds the value associated with key `IBMQ_TOKEN` as the IBM Quantum Token.

## Side Effects
The scripts will create folders `{DATA_FOLDER_NAME}/{DENSITY_MAT_FOLDER_NAME}`, `{DATA_FOLDER_NAME}/{GEN_CIRCS_FOLDER_NAME}`, and `{DATA_FOLDER_NAME}/{RESULTS_FOLDER_NAME}` for storing density matrices, generated circuits, and the results from optimization, respectively. 

## Repository Structure
- [**`README.md`**](README.md): Repository readme with setup and execution instructions.
- [**`baseline_script.py`**](baseline_script.py): Python script for executing the baseline technique on datasets.
- [**`enqode_script.py`**](enqode_script.py): Python script for executing the EnQode technique on datasets.
- [**`requirements.txt`**](requirements.txt): Requirements to be installed before running the Python scripts.
- [**`noise_model_ibm_brisbane.pkl`**](noise_model_ibm_brisbane.pkl): The noise model used in the scripts for noisy simulation.
- [**`sampled_dataset_indices.csv`**](sampled_dataset_indices.csv): The indices of the datapoints used when evaluating the baseline and EnQode technique in the paper.
- [**`utils.py`**](utils.py): Python file for utility functions that are shared between the baseline and EnQode techniques.

## Copyright
Copyright © 2025 Positive Technology Lab. All rights reserved. For permissions, contact ptl@rice.edu.