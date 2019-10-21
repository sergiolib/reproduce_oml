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


def gen_sine_data(n_functions=10, sample_length=320, samples_per_function=32):
    """
    Generate synthetic Incremental Sine Waves as defined in section 4.1
    :param n_id: number of functions to use (default is 10 as defined in the paper)
    :param n_samples: number of samples per function (default is 320 as defined in the paper)
    :return:
    """
    # Define how many sine functions to generate
    indices = sample(range(n_functions), n_functions)  # IDs in random order, use = [range(n_functions)] to sort them

    # Sample amplitude for each function
    amplitude = np.random.uniform(amp_min, amp_max, size=n_functions)
    # Sample phase for each function
    phase = np.random.uniform(phase_min, phase_max, n_functions)
    # Sample z used as inputs of the sine functions
    list_of_z = np.random.uniform(z_min, z_max, size=(n_functions, samples_per_function, sample_length))

    # Initialize a 3D array of functions x samples_per_function x sample_length
    y = np.zeros(shape=(n_functions, samples_per_function, sample_length))
    
    # Initialize a 1D array of samples x functions for z and k values as well
    x = np.zeros(shape=(n_functions, samples_per_function, sample_length, n_functions + 1))

    # For every function sample samples_per_function instances of length sample_length
    for i, ii in enumerate(indices):
        for j in range(samples_per_function):
            y[i, j, :] = np.sin(list_of_z[i, j] + phase[i]) * amplitude[i]
            x[i, j, :, 0] = list_of_z[i, j]
        x[i, :, :, 1 + ii] = 1

    return x, y
