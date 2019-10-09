import tensorflow as tf


def mrcl_isw_rln(inputs, n_layers=6, hidden_units_per_layer=300):
    """
    Representation learning network for the incremental sine waves dataset.
    :param inputs: Input placeholder
    :type inputs: tf.keras.Input
    :param n_layers: Number of hidden layers in the RLN
    :type n_layers: int
    :param hidden_units_per_layer: Number of units in each hidden layer
    :type hidden_units_per_layer: int
    :return: Representation of shape [n_samples, hidden_units_per_layer]
    :rtype: tf.Tensor
    """
    h = inputs
    for i in range(n_layers):
        h = tf.keras.layers.Dense(hidden_units_per_layer, activation='relu')(h)
    return h


def mrcl_isw_tln(inputs, n_layers=2, hidden_units_per_layer=300):
    """
    Task learning network for the incremental sine waves dataset.
    :param inputs: Input placeholder
    :type inputs: tf.keras.Input
    :param n_layers: Number of layers in the TLN. Last one is 1 unit which is linear.
    :type n_layers: int
    :param hidden_units_per_layer: Number of units in each hidden layer
    :type hidden_units_per_layer: int
    :return: Output of shape [n_samples, 1]
    :rtype: tf.Tensor
    """
    y = inputs
    for i in range(n_layers - 1):
        y = tf.keras.layers.Dense(hidden_units_per_layer, activation='relu')(y)
    y = tf.keras.layers.Dense(1)(y)
    return y


def mrcl_isw(inputs, n_layers_rln=6, n_layers_tln=2, hidden_units_per_layer=300, one_hot_depth=400):
    """
    Full MRCL model in section 4.2 of the paper. Predicts incremental sine waves.
    :param inputs: Input placeholder, a concatenation of z and the one hot encoding of k
    :type inputs: tf.keras.Input
    :param n_layers_rln: Number of layers in the RLN
    :type n_layers_rln: int
    :param n_layers_tln: Number of layers in the TLN
    :type n_layers_tln: int
    :param hidden_units_per_layer: Number of hidden units in each layer
    :type hidden_units_per_layer: int
    :param one_hot_depth: Length of the one hot encoding vectors
    :type one_hot_depth: int
    """
    h = mrcl_isw_rln(inputs, n_layers_rln, hidden_units_per_layer)
    y = mrcl_isw_tln(h, n_layers_tln, hidden_units_per_layer)
    return y
