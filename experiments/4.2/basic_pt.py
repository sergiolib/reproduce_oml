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
