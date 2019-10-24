import pytest
import tensorflow as tf
from random import random, randint
import sys
import numpy as np
sys.path.append("../../")


def test_isw_forward_pass_with_random_samples():
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
                                                           hidden_units_per_layer=300, one_hot_depth=one_hot_depth))
    h, y = model(x)
    assert y.shape == (1, 1)

def test_isw_forward_pass_with_training_set():
    from isw import mrcl_isw
    from datasets import synth_datasets
    from random import randint
    sine_waves_z, sine_waves_k, sine_waves_y = synth_datasets.gen_sine_data(n_id=900, n_samples=320)
    # Take :400 sequences for pretraining and 400: for evaluation
    pretraining_samples = {
        "z": sine_waves_z[:400 * 320],
        "k": sine_waves_k[:400 * 320],
        "y": sine_waves_y[:400 * 320]
    }
    assert pretraining_samples["k"][-1] == 399
    i = randint(0, len(pretraining_samples["k"]) - 1)
    one_hot_depth = 900
    k = tf.one_hot(pretraining_samples["k"], one_hot_depth)
    assert len(k) == len(pretraining_samples["z"]) == len(pretraining_samples["y"])
    assert np.where(k[i])[0][0] == pretraining_samples["k"][i]
    pretraining_samples["k"] = k
    pretraining_samples["z"] = tf.convert_to_tensor(pretraining_samples["z"].astype(np.float32))
    pretraining_samples["y"] = tf.convert_to_tensor(pretraining_samples["y"].astype(np.float32))
    pretraining_samples["z"] = tf.reshape(pretraining_samples["z"], (-1, 1))
    x = tf.concat([pretraining_samples["z"], pretraining_samples["k"]], axis=1)
    assert x.shape == (400 * 320, one_hot_depth + 1)
    inputs = tf.keras.Input(shape=(one_hot_depth + 1,))
    model = tf.keras.Model(inputs=inputs, outputs=mrcl_isw(inputs, n_layers_rln=6, n_layers_tln=2, 
                                                           hidden_units_per_layer=300, one_hot_depth=one_hot_depth))
    h, y_hat = model(x)
    assert len(y_hat) == 400 * 320
