import tensorflow as tf


def basic_pt_isw(n_layers=8, hidden_units_per_layer=300, one_hot_depth=400):
    input_nn = tf.keras.Input(shape=one_hot_depth + 1)
    h = input_nn
    # Define hidden layers
    for i in range(n_layers - 1):
        h = tf.keras.layers.Dense(hidden_units_per_layer, activation='relu')(h)

    # Define output layer
    y = tf.keras.layers.Dense(1)(h)

    # Define model
    basic_pt = tf.keras.Model(inputs=input_nn, outputs=y)
    return basic_pt


def basic_pt_omniglot(inputs, n_layers=8, hidden_units_per_layer=256):
    # They state that "Rather than restricting to the same 6-2 architecture for
    # the RLN and TLN, we pick the best split using a validation set."
    # But dont you need to at least know how many Convolutional layers and how many fully connected you have?
    # Are you supposed to grid-search all possible architectures?
    raise NotImplementedError

import tensorflow as tf
import sys
sys.path.append("../../datasets")
sys.path.append("../")
from synth_datasets import gen_sine_data, partition_sine_data
from training import split_data_in_2, mrcl_pretrain, mrcl_evaluate, basic_pt_train,  basic_pt_eval
from isw import mrcl_isw
from basic_pt import basic_pt_isw
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
partition = partition_sine_data(gen_sine_data(n_id=900))
pretraining = partition["pretraining"]
evaluation = partition["evaluation"]

pretraining["z"] = tf.convert_to_tensor(pretraining["z"], dtype=tf.float32)
pretraining["k"] = tf.convert_to_tensor(pretraining["k"], dtype=tf.int32)
pretraining["y"] = tf.convert_to_tensor(pretraining["y"], dtype=tf.float32)

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
    input_data["k"] = tf.convert_to_tensor(k_seq)


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


def prepare_train_data(with_split=True):
    # for rs in range(1000):
    # Sample 10 functions from the 400
    f_inds = np.random.permutation(range(400))[:20]
    f_inds_tr = f_inds[:10]
    f_inds_ev = f_inds[10:]
    train_data = concat_dicts([data_with_k_equals(pretraining, k) for k in f_inds_tr])
    reassign_k_values(train_data, f_inds_tr)
    train_data = do_one_hot(train_data)
    train_data = join_z_and_k(train_data)
    if with_split:
        s_learn, s_remember = split_data_in_2(train_data, 0.8)
        return f_inds_ev, s_learn, s_remember
    else:
        return f_inds_ev, train_data

def prepare_eval_data(f_inds_ev):
    eval_data = concat_dicts([data_with_k_equals(pretraining, k) for k in f_inds_ev])
    reassign_k_values(eval_data, f_inds_ev)
    eval_data = do_one_hot(eval_data)
    eval_data = join_z_and_k(eval_data)
    return eval_data

def run_mrcl():
    f_inds_ev, s_learn, s_remember = prepare_train_data()
    mrcl_pretrain((s_learn["x"], s_learn["y"]), (s_remember["x"], s_remember["y"]), rln, tln, regression_parameters)

    eval_data = prepare_eval_data(f_inds_ev)
    results = mrcl_evaluate(eval_data, rln, tln, regression_parameters)

    np.savetxt("eval_results.txt", results)
    np.savetxt("ground_truth.txt", np.array(eval_data["y"]))


def run_scratch():
    """
    Learn MRCL online without any pre-training

    TODO: Does from scratch mean when online training, update the whole network or just as MRCL, update only TLN?

    """
    f_inds_ev, _, _ = prepare_train_data()
    eval_data = prepare_eval_data(f_inds_ev)

    # make sure network starts with a random initialization
    results = mrcl_evaluate(eval_data, rln, tln, regression_parameters)
    np.savetxt("scratch_eval_results.txt", results)
    np.savetxt("scratch_ground_truth.txt", np.array(eval_data["y"]))


def run_basic_pt():
    """
    Use a very simple pre-training procedure of just training a single NN with Gradient Descent
    """

    params = {
        "learning_rate": 1e-3,
        "loss_metric": tf.losses.MSE,
        "total_gradient_updates": 40,
        "inner_steps": 400,  # k
        "optimizer": tf.compat.v1.train.GradientDescentOptimizer,
        "random_batch_size": 8  # len(X_rand)
    }

    n_layers = 8

    f_inds_ev, train_data = prepare_train_data(with_split=False)

    basic_model = basic_pt_isw(n_layers=n_layers)

    basic_pt_train(params, train_data, basic_model)

    eval_data = prepare_eval_data(f_inds_ev)

    # Approach no.1: Create and pretrain one model and then make different versions by splitting it in the evaluation
    # Approach no.2: Create and pretrain different models of RLN & TLN so you dont have to worry about different splits

    # Validation in order to find the best split
    # First layer is always fixed, last layer is always trainable
    # First freeze the first layer, then the first and second layer etc
    for i in range(1, n_layers - 2):
        results = basic_pt_eval(eval_data, basic_model, params, layers_to_freeze=i)
        np.savetxt(f"basic_pt_{i}_eval_results.txt", results)

    np.savetxt("ground_truth.txt", np.array(eval_data["y"]))


def run_srnn():
    pass


run_basic_pt()
