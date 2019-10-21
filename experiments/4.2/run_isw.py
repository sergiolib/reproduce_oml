import os

import tensorflow as tf
import sys

import tqdm

sys.path.append("../../datasets")
sys.path.append("../")
from synth_datasets import gen_sine_data
from training import split_data_in_2, mrcl_pretrain, mrcl_evaluate
from isw import mrcl_isw
import numpy as np


regression_parameters = {
    "meta_learning_rate": 1e-4,  # alpha
    "inner_learning_rate": 3e-3,  # beta
    "loss_metric": tf.losses.MSE,
    "total_gradient_updates": 40,  # n # TODO: check this as it is not explicit in the paper
    "inner_steps": 400,  # k
    "online_optimizer": tf.optimizers.SGD,
    "meta_optimizer": tf.optimizers.Adam,
    "random_batch_size": 8  # len(X_rand)
}

rln, tln = mrcl_isw(one_hot_depth=10)  # Actual model


def data_with_k_equals(pretraining, k):
    """Return the dict of data of only task k"""
    new_data_dict = {}
    for s in pretraining:
        new_data_dict[s] = pretraining[s][pretraining["k"] == k]
    return new_data_dict


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


trange = tqdm.trange(40)
for rs in trange:
    # Sample 10 functions from the 400 for this pretraining
    data = gen_sine_data(n_functions=10)

    pretraining = {}

    pretraining["x"] = tf.convert_to_tensor(data[0], dtype=tf.float32)
    pretraining["y"] = tf.convert_to_tensor(data[1], dtype=tf.float32)

    # Prepare data input
    s_learn, s_remember = split_data_in_2(pretraining, 0.8)
    # Pretrain on
    mrcl_pretrain((s_learn["x"], s_learn["y"]), (s_remember["x"], s_remember["y"]), rln, tln, regression_parameters)

    # non_sparse_vals = np.count_nonzero(rln(train_data["x"]), axis=1)
    # trange.set_description(f"Non sparse elements: {np.mean(non_sparse_vals / 300)}")

    if rs % 10 == 0:
        save_models(rs, rln, tln)

save_models("final", rln, tln)

mean_error_dist = np.zeros(10)
eval_iters = 10
for rs in range(eval_iters):
    f_inds_ev = eval_fs[rs * 10:rs * 10 + 10]
    eval_data = concat_dicts([data_with_k_equals(evaluation, k) for k in f_inds_ev])
    reassign_k_values(eval_data, f_inds_ev)
    eval_data = do_one_hot(eval_data)
    eval_data = join_z_and_k(eval_data)
    seen_data, unseen_data = split_data_in_2(eval_data, 0.8)
    results = mrcl_evaluate(seen_data, rln, tln, regression_parameters)
    np.savetxt(f"loss_results_{rs}.txt", results)
    mean_error_dist += calculate_error_distribution(unseen_data, rln, tln, regression_parameters["loss_metric"])
mean_error_dist /= eval_iters

np.savetxt("error_distribution.txt", mean_error_dist)

print(mean_error_dist)

