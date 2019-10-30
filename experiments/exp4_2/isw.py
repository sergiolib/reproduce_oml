import tensorflow as tf


def mrcl_isw_rln(inputs, n_layers=6, hidden_units_per_layer=300, representation_size=900, seed=None):
    """
    Representation learning network for the incremental sine waves dataset.
    :param inputs: Input placeholder
    :type inputs: tf.keras.Input
    :param n_layers: Number of hidden layers in the RLN
    :type n_layers: int
    :param representation_size: Number of outputs from RLN
    :type representation_size: int
    :param hidden_units_per_layer: Number of units in each hidden layer
    :type hidden_units_per_layer: int
    :return: Representation of shape [n_samples, representation_size]
    :rtype: tf.Tensor
    """
    h = inputs
    initializer = tf.keras.initializers.he_normal(seed=seed)
    for i in range(n_layers - 1):
        h = tf.keras.layers.Dense(hidden_units_per_layer, activation='relu', kernel_initializer=initializer)(h)
    h = tf.keras.layers.Dense(representation_size, activation='relu', kernel_initializer=initializer)(h)
    return h


def mrcl_isw_tln(inputs, n_layers=2, hidden_units_per_layer=300, seed=None):
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
    initializer = tf.keras.initializers.he_normal(seed=seed)
    for i in range(n_layers):
        y = tf.keras.layers.Dense(hidden_units_per_layer, activation='relu', kernel_initializer=initializer)(y)
    y = tf.keras.layers.Dense(1, kernel_initializer="he_normal")(y)
    return y


def mrcl_isw(n_layers_rln=6, n_layers_tln=2, hidden_units_per_layer=300, one_hot_depth=10, representation_size=900, seed=0):
    """
    Full MRCL model in section exp4_2 of the paper. Predicts incremental sine waves.
    :param seed: Random seed for fixing initializations
    :param n_layers_rln: Number of layers in the RLN
    :type n_layers_rln: int
    :param n_layers_tln: Number of layers in the TLN
    :type n_layers_tln: int
    :param representation_size: Number of outputs from RLN
    :type representation_size: int
    :param hidden_units_per_layer: Number of hidden units in each layer
    :type hidden_units_per_layer: int
    :param one_hot_depth: Length of the one hot encoding vectors
    :type one_hot_depth: int
    :rtype: (tf.Tensor, tf.Tensor)
    """
    input_rln = tf.keras.Input(shape=one_hot_depth + 1)
    input_tln = tf.keras.Input(shape=representation_size)
    h = mrcl_isw_rln(input_rln, n_layers_rln, hidden_units_per_layer, seed=seed)
    rln = tf.keras.Model(inputs=input_rln, outputs=h)
    y = mrcl_isw_tln(input_tln, n_layers_tln, hidden_units_per_layer, seed=seed)
    tln = tf.keras.Model(inputs=input_tln, outputs=y)
    return rln, tln
