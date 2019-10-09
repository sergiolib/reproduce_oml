import pytest
import tensorflow as tf
from random import random, randint


def test_isw_with_random_samples():
    from isw import mrcl_isw
    random_z = (random()-0.5) * 10
    random_k = randint(0, 9)
    one_hot_depth = 400
    random_k = tf.one_hot(random_k, one_hot_depth)
    x = tf.concat([[random_z], random_k], axis=0)
    assert x.shape == (401)
    x = tf.reshape(x, (1, -1))
    assert x.shape == (1, 401)
    inputs = tf.keras.Input(shape=(one_hot_depth + 1,))
    model = tf.keras.Model(inputs=inputs, outputs=mrcl_isw(inputs, n_layers_rln=6, n_layers_tln=2, 
                                                           hidden_units_per_layer=300, one_hot_depth=400))
    y = model(x)
    assert y.shape == (1, 1)
