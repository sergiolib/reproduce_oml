import datetime
import os
from shutil import rmtree

import tensorflow as tf
import sys
import numpy as np
import tqdm

from math import pi
from random import sample

sys.path.append("experiments/4.2/")
from isw import mrcl_isw

regression_parameters = {
    "inner_learning_rate": 3e-3,  # beta
    "meta_learning_rate": 1e-4,  # beta
    "total_gradient_updates": 40,  # n
    "inner_batch_size": 400,  # len(X_traj)
    "inner_steps": 400,  # k
    "online_optimizer": tf.optimizers.SGD,
    "random_batch_size": 8  # len(X_rand)
}

def data_with_k_equals(pretraining, k):
    """Return the dict of data of only task k"""
    new_data_dict = {}
    for s in pretraining:
        new_data_dict[s] = pretraining[s][pretraining["k"] == k]
    return new_data_dict


def copy_parameters(source, dest):
    """Copies parameters of source model into the destination one"""
    for matrix_source, matrix_dest in zip(source.weights, dest.weights):
        matrix_dest.assign(matrix_source)


def concat_dicts(list_of_dicts):
    """Concat the dicts of data"""
    new_data_dict = {}
    for d in list_of_dicts:
        for s in d:
            new_data_dict[s] = tf.concat([new_data_dict[s], d[s]], axis=-1) if s in new_data_dict else d[s]
    return new_data_dict


def reassign_k_values(input_data, f_inds):
    """Reassign the values of k in the order provided in f_inds"""
    k_seq = np.array(input_data["k"])
    for new_k, old_k in enumerate(f_inds):
        k_seq[k_seq == old_k] = new_k
    input_data["k"] = tf.convert_to_tensor(k_seq, dtype=tf.int32)


def join_z_and_k(input_data):
    """Combine z and k streams into x"""
    z = input_data["z"]
    k = input_data["k"]
    x = tf.concat([tf.reshape(z, (-1, 1)), k], axis=1)
    del input_data["z"]
    del input_data["k"]
    input_data["x"] = x
    return input_data


def do_one_hot(input_data):
    """Transform the k values into one hot encoding"""
    k = input_data["k"]
    k = tf.one_hot(k, depth=10)
    input_data["k"] = k
    return input_data


def save_models(rs):
    try:
        os.path.isdir("saved_models/")
    except NotADirectoryError:
        os.makedirs("save_models")
    rln.save(f"saved_models/rln_pretraining_{rs}.tf", save_format="tf")
    tln.save(f"saved_models/tln_pretraining_{rs}.tf", save_format="tf")


def calculate_error_distribution(unseen_data, rln, tln, loss_fun, n_tasks=10,):
    classes = np.argmax(unseen_data["x"][:, 1:], axis=1)
    losses = loss_fun(tln(rln(unseen_data["x"])), unseen_data["y"])
    hist = np.zeros(n_tasks)
    _, n_elems_per_class = np.unique(np.argmax(unseen_data["x"][:, 1:], axis=1), return_counts=True)
    for i, l in enumerate(losses):
        t = classes[i]
        hist[t] += l
    return hist / n_elems_per_class

def sample_trajectory_generators(s_learn, max_samples):
    inds_id = np.arange(len(s_learn["x"]))
    seqs_id = np.arange(len(s_learn["x"][0]))
    samples_id = np.arange(len(s_learn["x"][0][0]))
    prod = list(product(inds_id, seqs_id, samples_id))
    last_sample = 0
    while True:
        np.random.shuffle(prod)
        samples_x, samples_y = [], []
        while len(samples_x) < max_samples:
            i, j, k = prod[last_sample % len(prod)]
            samples_x.append(s_learn["x"][i, j, k])
            samples_y.append(s_learn["y"][i, j, k])
            last_sample += 1

        samples_x = np.vstack(samples_x)
        samples_y = np.hstack(samples_y)

        yield samples_x, samples_y

# @tf.function
def inner_update(x, y):
    with tf.GradientTape(watch_accessed_variables=False) as Wj_Tape:
        Wj_Tape.watch(tln.trainable_variables)
        inner_loss = compute_loss(x, y)
    gradients = Wj_Tape.gradient(inner_loss, tln.trainable_variables)
    meta_optimizer_inner.apply_gradients(zip(gradients, tln.trainable_variables))

#@tf.function
def compute_loss(x, y):
    output = tln(rln(x))
    loss = loss_fun(output, y)
    return loss

# Constant Sine values as defined in section 4.1
amp_min = 0.1
amp_max = 5
phase_min = 0
phase_max = pi
z_min = -5
z_max = 5

def gen_sine_data(tasks=None, n_functions=10, sample_length=32, repetitions=40):
    """
    Generate synthetic Incremental Sine Waves as defined in section 4.1
    :param n_id: number of functions to use (default is 10 as defined in the paper)
    :param sample_length: number of samples per function (default is 32 as defined in the paper)
    :return:
    """
    if tasks is None:
        tasks = {}
        # Sample amplitude for each function
        tasks["amplitude"] = np.random.uniform(amp_min, amp_max, size=400)
        # Sample phase for each function
        tasks["phase"] = np.random.uniform(phase_min, phase_max, 400)
    subsample = sample(list(zip(tasks["amplitude"], tasks["phase"])), 10)
    amplitude = [s[0] for s in subsample]
    phase = [s[1] for s in subsample]
    # Sample z used as inputs of the sine functions
    list_of_z = np.random.uniform(z_min, z_max, size=(n_functions, repetitions, sample_length))

    y_traj = np.zeros(shape=(n_functions, repetitions, sample_length))
    y_rand = np.zeros(shape=(n_functions, sample_length))

    x_traj = np.zeros(shape=(n_functions, repetitions, sample_length, n_functions + 1))
    x_rand = np.zeros(shape=(n_functions, sample_length, n_functions + 1))

    # For every function sample samples_per_function instances of length sample_length
    for ind in range(10):
        for repetition in range(repetitions):
            y_traj[ind, repetition, :] = np.sin(list_of_z[ind, repetition] + phase[ind]) * amplitude[ind]
            x_traj[ind, repetition, :, 0] = list_of_z[ind, repetition]
            x_traj[ind, repetition, :, 1 + ind] = 1
            
    list_of_z = np.random.uniform(z_min, z_max, size=(n_functions, sample_length))
            
    for ind in range(10):
        y_rand[ind, :] = np.sin(list_of_z[ind, 0] + phase[ind]) * amplitude[ind]
        x_rand[ind, :, 0] = list_of_z[ind]
        x_rand[ind, :, 1 + ind] = 1

    return x_traj, y_traj, x_rand, y_rand, tasks

rln, tln = mrcl_isw(one_hot_depth=10)  # Actual model
loss_fun = tf.keras.losses.MeanSquaredError()
meta_optimizer_inner = tf.keras.optimizers.SGD(learning_rate=regression_parameters["inner_learning_rate"])
meta_optimizer_outer = tf.keras.optimizers.Adam(learning_rate=regression_parameters["meta_learning_rate"])

def pretrain_mrcl(x_traj, y_traj, x_rand, y_rand):
    # Random reinitialization of last layer
    w = tln.layers[-1].weights[0]
    new_w = tln.layers[-1].kernel_initializer(shape=w.shape)
    tln.layers[-1].weights[0].assign(new_w)

    # Clone tln to preserve initial weights
    tln_initial = tf.keras.models.clone_model(tln)

    # Sample x_rand, y_rand from s_remember
    x_traj_f = tf.concat([x for x in x_traj], axis=0)
    y_traj_f = tf.concat([y for y in y_traj], axis=0)

    x_meta = tf.concat([x_rand, x_traj_f], axis=0)
    y_meta = tf.concat([y_rand, y_traj_f], axis=0)

    for x, y in tf.data.Dataset.from_tensor_slices((x_traj, y_traj)):
        inner_update(x, y)

    with tf.GradientTape(persistent=True) as theta_Tape:
        outer_loss = compute_loss(x_meta, y_meta)

    tln_gradients = theta_Tape.gradient(outer_loss, tln.trainable_variables)
    rln_gradients = theta_Tape.gradient(outer_loss, rln.trainable_variables)
    del theta_Tape
    meta_optimizer_outer.apply_gradients(zip(tln_gradients + rln_gradients, tln_initial.trainable_variables + rln.trainable_variables))

    copy_parameters(tln_initial, tln)

    return outer_loss

t = tqdm.trange(20000)
tasks = None

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
rmtree('logs')
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

for epoch, v in enumerate(t):
    x_traj, y_traj, x_rand, y_rand, tasks = gen_sine_data(tasks=tasks, n_functions=10, sample_length=32, repetitions=40)
    
    x_traj = np.vstack(x_traj)

    y_traj = np.vstack(y_traj)

    x_rand = np.vstack(x_rand)

    y_rand = np.hstack(y_rand)

    x_rand = tf.convert_to_tensor(x_rand, dtype=tf.float32)
    y_rand = tf.convert_to_tensor(y_rand, dtype=tf.float32)
    x_traj = tf.convert_to_tensor(x_traj, dtype=tf.float32)
    y_traj = tf.convert_to_tensor(y_traj, dtype=tf.float32)
    loss = pretrain_mrcl(x_traj, y_traj, x_rand, y_rand)
    
    # Check metrics
    rep = rln(x_rand)
    rep = np.array(rep)
    counts = np.isclose(rep, 0).sum(axis=1) / rep.shape[1]
    sparsity = np.mean(counts)
    with train_summary_writer.as_default():
        tf.summary.scalar('Sparsity', sparsity, step=epoch)
        tf.summary.scalar('Training loss', loss, step=epoch)
    t.set_description(f"Sparsity: {sparsity}\tTraining loss: {loss}")

    if epoch % 100 == 0 and epoch > 0:
        save_models(epoch)
