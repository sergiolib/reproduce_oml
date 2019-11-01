import tensorflow as tf

from os import makedirs
from os.path import isdir

from datasets.synth_datasets import gen_sine_data
import numpy as np


def copy_parameters(source, dest):
    for s, d in zip(source.trainable_variables, dest.trainable_variables):
        d.assign(s)


def save_models(epoch, rln, tln):
    try:
        isdir("saved_models/")
    except NotADirectoryError:
        makedirs("save_models")
    rln.save(f"saved_models/rln_pretraining_{epoch}.tf", save_format="tf")
    tln.save(f"saved_models/tln_pretraining_{epoch}.tf", save_format="tf")


def save_models(model, name):
    try:
        isdir("saved_models/")
    except NotADirectoryError:
        makedirs("saved_models/")
    model.save(f"saved_models/{name}.tf", save_format="tf")


@tf.function
def inner_update(x, y, tln, rln, beta, loss_fun):
    with tf.GradientTape(watch_accessed_variables=False) as Wj_Tape:
        Wj_Tape.watch(tln.trainable_variables)
        inner_loss = compute_loss(x, y, tln=tln, rln=rln, loss_fun=loss_fun)
    gradients = Wj_Tape.gradient(inner_loss, tln.trainable_variables)
    for g, v in zip(gradients, tln.trainable_variables):
        v.assign(v - beta * g)


@tf.function
def compute_loss(x, y, tln, rln, loss_fun):
    output = tln(rln(x))
    loss = loss_fun(output, y)
    return loss


def pretrain_mrcl(x_traj, y_traj, x_rand, y_rand, tln, tln_initial, rln, meta_optimizer, loss_function, beta,
                  reset_last_layer=True):
    if reset_last_layer:
        # Random reinitialization of last layer
        last_layer = tln.layers[-1]
        w = last_layer.weights[0]
        b = last_layer.weights[1]

        new_w = last_layer.kernel_initializer(shape=w.shape)

        zero_initializer = tf.keras.initializers.zeros()
        new_b = zero_initializer(shape=b.shape)

        w.assign(new_w)
        b.assign(new_b)

    # Save actual values for later retrieval
    copy_parameters(tln, tln_initial)

    # Sample x_rand, y_rand from s_remember
    x_shape = x_traj.shape
    x_traj_f = tf.reshape(x_traj, (x_shape[0] * x_shape[1], x_shape[2]))
    y_traj_f = tf.reshape(y_traj, (x_shape[0] * x_shape[1],))

    x_meta = tf.concat([x_rand, x_traj_f], axis=0)
    y_meta = tf.concat([y_rand, y_traj_f], axis=0)

    for x, y in tf.data.Dataset.from_tensor_slices((x_traj, y_traj)):
        inner_update(x=x, y=y, tln=tln, rln=rln, beta=beta,
                     loss_fun=loss_function)

    with tf.GradientTape(persistent=True) as theta_Tape:
        outer_loss = compute_loss(x=x_meta, y=y_meta, tln=tln, rln=rln, loss_fun=loss_function)

    tln_gradients = theta_Tape.gradient(outer_loss, tln.trainable_variables)
    rln_gradients = theta_Tape.gradient(outer_loss, rln.trainable_variables)
    del theta_Tape
    meta_optimizer.apply_gradients(zip(tln_gradients + rln_gradients,
                                       tln_initial.trainable_variables + rln.trainable_variables))

    # Retrieve updated tln parameters
    copy_parameters(tln_initial, tln)

    return outer_loss

def prepare_data_pre_training(tasks, n_functions, sample_length, repetitions):
    # Sample data
    x_traj, y_traj, x_rand, y_rand = gen_sine_data(tasks=tasks,
                                                   n_functions=n_functions,
                                                   sample_length=sample_length,
                                                   repetitions=repetitions)

    # Reshape for inputting to training method
    x_traj = np.vstack(x_traj)
    y_traj = np.vstack(y_traj)
    x_rand = np.vstack(x_rand)
    y_rand = np.hstack(y_rand)

    # Numpy -> Tensorflow
    x_rand = tf.convert_to_tensor(x_rand, dtype=tf.float32)
    y_rand = tf.convert_to_tensor(y_rand, dtype=tf.float32)
    x_traj = tf.convert_to_tensor(x_traj, dtype=tf.float32)
    y_traj = tf.convert_to_tensor(y_traj, dtype=tf.float32)

    return x_traj, y_traj, x_rand, y_rand


def to_iid(x1, y1, n_functions, sample_length, repetitions):
    # Reshape to 2D
    x1_t = tf.reshape(x1, [-1, n_functions + 1])
    y1_t = tf.reshape(y1, [-1,])

    indexes = np.random.permutation(len(x1_t))

    x1_t = tf.gather(x1_t, indexes)
    y1_t = tf.gather(y1_t, indexes)

    # Reshape back
    x1 = tf.reshape(x1_t, [10 * repetitions, sample_length, n_functions + 1])
    y1 = tf.reshape(y1_t, [10 * repetitions, sample_length])

    return x1, y1
