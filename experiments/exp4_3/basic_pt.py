import tensorflow as tf
import numpy as np


def run_isw():
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


def run_omniglot():
    pass


def basic_pt_isw_model(n_layers=8, hidden_units_per_layer=300, one_hot_depth=400):
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


def basic_pt_omniglot_model(inputs, n_layers=8, hidden_units_per_layer=256):
    # They state that "Rather than restricting to the same 6-2 architecture for
    # the RLN and TLN, we pick the best split using a validation set."
    # But dont you need to at least know how many Convolutional layers and how many fully connected you have?
    # Are you supposed to grid-search all possible architectures?
    raise NotImplementedError
