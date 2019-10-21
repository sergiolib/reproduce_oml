import os

import tensorflow as tf
import sys
import numpy as np
import tqdm

sys.path.append("datasets/")
from synth_datasets import gen_sine_data

sys.path.append("experiments/")
from training import split_data_in_2#, mrcl_pretrain, mrcl_evaluate

sys.path.append("experiments/4.2/")
from isw import mrcl_isw

regression_parameters = {
    "inner_learning_rate": 3e-3,  # beta
    "total_gradient_updates": 40,  # n
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


def save_models(rs, rln, tln):
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

@tf.function
def inner_update(x, y):
    with tf.GradientTape(watch_accessed_variables=False) as Wj_Tape:
        Wj_Tape.watch(tln.trainable_variables)
        inner_loss = compute_loss(x, y)
    gradients = Wj_Tape.gradient(inner_loss, tln.trainable_variables)
    meta_optimizer_inner.apply_gradients(zip(gradients, tln.trainable_variables))

@tf.function
def compute_loss(x, y):
    output = tln(rln(x))
    loss = loss_fun(output, y)
    return loss


def mrcl_pretrain(s_learn, s_remember):
    #for i in range(params["total_gradient_updates"]):  # each of the 40 optimizations
    for i in range(regression_parameters["total_gradient_updates"]):
        i_learn = iter(s_learn)
        i_rem = iter(s_remember)
        # Random reinitialization of last layer
        w = tln.layers[-1].weights[0]
        new_w = tln.layers[-1].kernel_initializer(shape=w.shape)
        tln.layers[-1].weights[0].assign(new_w)

        # Clone tln to preserve initial weights
        tln_initial = tf.keras.models.clone_model(tln)

        with tf.GradientTape(watch_accessed_variables=False) as W_Tape:
            W_Tape.watch(tln.trainable_variables)
            for x_traj, y_traj in i_learn:
                # with tf.GradientTape(watch_accessed_variables=False) as Wj_Tape:
                #     Wj_Tape.watch(tln.trainable_variables)
                #     inner_loss = compute_loss(tf.reshape(x_traj[j], (1, -1)), y_traj[j], rln, tln, loss_fun)
                # gradients = Wj_Tape.gradient(inner_loss, tln.trainable_variables)
                # meta_optimizer_inner.apply_gradients(zip(gradients, tln.trainable_variables))
                inner_update(x_traj, y_traj)

            # Sample x_rand, y_rand from s_remember
            x_rand, y_rand = next(i_rem)
            x_meta = tf.concat([x_rand, x_traj], axis=0)
            y_meta = tf.concat([y_rand, y_traj], axis=0)
            #inds = np.random.permutation(len(x_meta))
            #x_meta = tf.gather(x_meta, inds)
            #y_meta = tf.gather(y_meta, inds)

            with tf.GradientTape(watch_accessed_variables=False) as theta_Tape:
                theta_Tape.watch(rln.trainable_variables)
                outer_loss = compute_loss(x_meta, y_meta)

        tln_gradients = W_Tape.gradient(outer_loss, tln.trainable_variables)
        # prv = float(tln_initial.trainable_variables[0][0, 0])
        meta_optimizer_outer.apply_gradients(zip(tln_gradients, tln_initial.trainable_variables))
        # aft = float(tln_initial.trainable_variables[0][0, 0])
        # grd = float(tln_gradients[0][0, 0])
        # if grd > 0:
        #     assert aft != prv
        rln_gradients = theta_Tape.gradient(outer_loss, rln.trainable_variables)
        meta_optimizer_outer.apply_gradients(zip(rln_gradients, rln.trainable_variables))

        copy_parameters(tln_initial, tln)

rln, tln = mrcl_isw(one_hot_depth=10)  # Actual model
loss_fun = tf.keras.losses.MSE
meta_optimizer_inner = tf.keras.optimizers.SGD(learning_rate=1e-4)
meta_optimizer_outer = tf.keras.optimizers.Adam(learning_rate=1e-4)
data = gen_sine_data(n_functions=10)

for rs in range(40):
    # Sample 10 functions from the 400 for this pretraining
    data = gen_sine_data(n_functions=10)

    x = tf.convert_to_tensor(data[0], dtype=tf.float32)
    y = tf.convert_to_tensor(data[1], dtype=tf.float32)

    x_learn = x[:, :, :312]
    x_remember = x[:, :, 312:]

    y_learn = y[:, :, :312]
    y_remember = y[:, :, 312:]

    # Reshapes
    xp = tf.transpose(x_learn, [3, 0, 1, 2])
    lx = xp.shape[0]
    xpp = tf.reshape(xp, (lx, -1))
    ypp = tf.reshape(y_learn, [-1])
    xpp = tf.transpose(xpp)
    y_learn = ypp
    x_learn = xpp

    # Reshapes
    xp = tf.transpose(x_remember, [3, 0, 1, 2])
    lx = xp.shape[0]
    xpp = tf.reshape(xp, (lx, -1))
    ypp = tf.reshape(y_remember, [-1])
    xpp = tf.transpose(xpp)
    y_remember = ypp
    x_remember = xpp
    
    # Convert to tf.data.Dataset format
    s_learn = tf.data.Dataset.from_tensor_slices((x_learn, y_learn)).shuffle(len(x_learn)).batch(400)
    s_remember = tf.data.Dataset.from_tensor_slices((x_remember, y_remember)).shuffle(len(x_remember)).batch(400)
        
    # Pretrain on
    mrcl_pretrain(s_learn, s_remember)

    non_sparse_vals = np.count_nonzero(rln(x_remember), axis=1)
    print(f"{rs}: Non sparse elements: {np.mean(non_sparse_vals / 300)}")

    if rs % 10 == 0 and rs > 0:
        save_models(rs, rln, tln)
