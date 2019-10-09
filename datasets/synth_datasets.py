""" Loader file for synthetic datasets"""

import numpy as np
from math import pi
from random import sample

# Constant Sine values as defined in section 4.1
amp_min = 0.1
amp_max = 5
phase_min = 0
phase_max = pi
z_min = -5
z_max = 5


def gen_sine_data(n_id=10, n_samples=320):
    """
    Generate synthetic Incremental Sine Waves as defined in section 4.1
    :param n_id: number of functions to use (default is 10 as defined in the paper)
    :param n_samples: number of samples per function (default is 320 as defined in the paper)
    :return:
    """
    # Define how many sine functions to generate
    indices = sample(range(n_id), n_id)  # IDs in random order, use = [range(n_id)] to sort them

    # Sample amplitude for each function
    amplitude = np.random.uniform(amp_min, amp_max, n_id)
    # Sample phase for each function
    phase = np.random.uniform(phase_min, phase_max, n_id)
    # Sample z used as inputs of the sine functions
    list_of_z = np.random.uniform(z_min, z_max, size=(n_samples, n_id))

    # Initialize a 1D array of samples x functions. The samples of each function are concatenated in sequence
    y = np.zeros(shape=(n_samples * n_id))
    
    # Initialize a 1D array of samples x functions for z and k values as well
    z = np.zeros(shape=(n_samples * n_id))
    k = np.zeros(shape=(n_samples * n_id), dtype=int)

    # For every function sample n_samples and add them to the correct indices
    for i in indices:
        start = i * n_samples
        end = start + n_samples
        y[start:end] = np.sin(list_of_z[:, i] + phase[i]) * amplitude[i]
        z[start:end] = list_of_z[:, i]
        k[start:end] = i

    return z, k, y

def partition_sine_data(sine_data, pretraining_n_seq=400, evaluation_n_seq=500, seq_len=320):
    """
    Partition the incremental sine waves dataset
    :param sine_data: Direct output of gen_sine_data
    :type sine_data: (numpy.ndarray, numpy.ndarray, numpy.ndarray)
    :param pretraining_n_seq: Number of sequences to include in pretraining set
    :type pretraining_n_seq: int
    :param evaluation_n_seq: Number of sequences to include in evaluation set
    :type evaluation_n_seq: int
    :param seq_len: Length of each sequence
    :type seq_len: int
    :return: Partitioned pretraining and evaluation sets
    :rtype: dict
    """
    partitioned = {}
    z, k, y = sine_data
    border_index = pretraining_n_seq * seq_len
    ptr_z = z[:border_index]
    eval_z = z[border_index:]
    ptr_k = k[:border_index]
    eval_k = k[border_index:]
    ptr_y = y[:border_index]
    eval_y = y[border_index:]
    pretraining = partitioned["pretraining"] = {}
    evaluation = partitioned["evaluation"] = {}
    pretraining["z"] = ptr_z
    pretraining["k"] = ptr_k
    pretraining["y"] = ptr_y
    evaluation["z"] = eval_z
    evaluation["k"] = eval_k
    evaluation["y"] = eval_y
    return partitioned
