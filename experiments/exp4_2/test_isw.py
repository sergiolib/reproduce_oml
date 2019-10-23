import tensorflow as tf


def test_isw_forward_pass_with_random_samples():
    from experiments.exp4_2.isw import mrcl_isw
    from random import random, randint
    rln, tln = mrcl_isw()
    random_z = (random() - 0.5) * 10
    random_k = randint(0, 9)
    one_hot_depth = 10
    random_k = tf.one_hot(random_k, one_hot_depth)
    x = tf.concat([[random_z], random_k], axis=0)
    assert x.shape == 11
    x = tf.reshape(x, (1, -1))
    assert x.shape == (1, 11)
    y = tln(rln(x))
    assert y.shape == (1, 1)


def test_isw_forward_pass_with_training_set():
    from experiments.exp4_2.isw import mrcl_isw
    from datasets import synth_datasets
    tasks = synth_datasets.gen_tasks(10)
    x_traj, y_traj, x_rand, y_rand = synth_datasets.gen_sine_data(tasks=tasks, n_functions=10, sample_length=32,
                                                                  repetitions=1)
    # Take :400 sequences for pre training and 400: for evaluation
    rln, tln = mrcl_isw()
    pred = tln(rln(x_traj[0, 0]))
    assert pred.shape == (32, 1)
