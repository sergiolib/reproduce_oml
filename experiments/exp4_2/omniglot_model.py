import tensorflow as tf

def mrcl_omniglot_rln(inputs, n_layers=6, filters=256, strides=[2,1,2,1,2,2]):
    h = inputs
    for i in range(n_layers):
        h = tf.keras.layers.Conv2D(filters, (3, 3), activation='relu', input_shape=(84, 84, 1), strides=strides[i])(h)
    h = tf.keras.layers.Flatten()(h)
    return h


def mrcl_omniglot_tln(inputs, n_layers=2, hidden_units_per_layer=300, output=964):
    y = inputs
    for i in range(n_layers - 1):
        y = tf.keras.layers.Dense(hidden_units_per_layer, activation='relu')(y)
    y = tf.keras.layers.Dense(output)(y)
    return y


def mrcl_omniglot():
    input_rln = tf.keras.Input(shape=(84, 84, 1))
    input_tln = tf.keras.Input(shape=3*3*256)
    h = mrcl_omniglot_rln(input_rln)
    rln = tf.keras.Model(inputs=input_rln, outputs=h)
    y = mrcl_omniglot_tln(input_tln)
    tln = tf.keras.Model(inputs=input_tln, outputs=y)
    return rln, tln