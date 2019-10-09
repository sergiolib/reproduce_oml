""" Loader file for synthetic datasets"""

import tensorflow as tf
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
    # Create N one-hot vectors, each indicating an ID for which sine function to use
    k = tf.one_hot(indices, n_id)

    # Sample amplitude for each function
    amplitude = np.random.uniform(amp_min, amp_max, n_id)
    # Sample phase for each function
    phase = np.random.uniform(phase_min, phase_max, n_id)
    # Sample z used as inputs of the sine functions
    list_of_z = np.random.uniform(z_min, z_max, size=(n_samples, n_id))

    # Initialize a 1D array of samples x functions. The samples of each function are concatenated in sequence
    y = np.zeros(shape=(n_samples * n_id))

    # For every function sample n_samples and add them to the correct indices
    for i in indices:
        start = i * n_samples
        end = start + n_samples
        y[start:end] = np.sin(list_of_z[:, i] + phase[i]) * amplitude[i]

    y = tf.convert_to_tensor(y)

    return y
