""" Loader file for synthetic data sets"""

import numpy as np
from math import pi
import random

# Constant Sine values as defined in section 4.1
amp_min = 0.1
amp_max = 5
phase_min = 0
phase_max = pi
z_min = -5
z_max = 5


def gen_tasks(number_of_tasks):
    """
    Generate tasks for the generation of sine waves
    :param number_of_tasks: Number of tasks to generate
    :type number_of_tasks: int
    :return: amplitude and phase generated samples for each possible task
    :rtype: dict
    """
    return {"amplitude": np.random.uniform(amp_min, amp_max, size=number_of_tasks),
            "phase": np.random.uniform(phase_min, phase_max, size=number_of_tasks)}


def gen_sine_data(tasks, n_functions=10, sample_length=32, repetitions=40, n_ids=10, seed=None):
    """
    Generate synthetic Incremental Sine Waves as defined in section 4.1
    :param seed: Seed
    :param tasks: amplitude and phase generated samples for each possible task
    :type tasks: dict
    :param n_functions: number of functions to use (default is 10 as defined in the paper)
    :type n_functions: int
    :param sample_length: number of samples per function (default is 32 as defined in the paper)
    :type sample_length: int
    :param repetitions: number of repetitions of each task
    :type repetitions: int
    :param n_ids: number of ids to generate
    :type n_ids: int
    :return: x trajectory samples, y trajectory samples, x random samples, y random samples
    :rtype numpy.ndarray (n_functions * repetitions x sample_length x n_ids),
           numpy.ndarray (n_functions * repetitions x sample_length),
           numpy.ndarray (n_functions x sample_length x n_ids),
           numpy.ndarray (n_functions x sample_length)
    """
    random.seed(seed)
    np.random.seed(seed)
    tasks_subsample = random.sample(list(zip(tasks["amplitude"], tasks["phase"])), n_functions)  # tuples: (amp, phase)
    amplitude = [task_parameters[0] for task_parameters in tasks_subsample]
    phase = [s[1] for s in tasks_subsample]

    # Sample z used as inputs of the sine functions
    list_of_z_traj = np.random.uniform(z_min, z_max, size=(n_functions, repetitions, sample_length))
    list_of_z_rand = np.random.uniform(z_min, z_max, size=(n_functions, sample_length))

    # Initialize input arrays
    x_traj = np.zeros(shape=(n_functions, repetitions, sample_length, n_ids + 1))
    x_rand = np.zeros(shape=(n_functions, sample_length, n_ids + 1))
    y_traj = np.zeros(shape=(n_functions, repetitions, sample_length))
    y_rand = np.zeros(shape=(n_functions, sample_length))

    # For every function, sample "repetitions" instances of length "sample_length"
    for ind in range(n_functions):
        for repetition in range(repetitions):
            y_traj[ind, repetition, :] = np.sin(list_of_z_traj[ind, repetition] + phase[ind]) * amplitude[ind]
            x_traj[ind, repetition, :, 0] = list_of_z_traj[ind, repetition]
            x_traj[ind, repetition, :, 1 + (ind // 10) % 10] = 1

        y_rand[ind, :] = np.sin(list_of_z_rand[ind] + phase[ind]) * amplitude[ind]
        x_rand[ind, :, 0] = list_of_z_rand[ind]
        x_rand[ind, :, 1 + (ind // 10) % 10] = 1

    return x_traj, y_traj, x_rand, y_rand
