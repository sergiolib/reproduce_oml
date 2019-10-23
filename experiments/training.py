import tensorflow as tf

from os import makedirs
from os.path import isdir

from experiments.exp4_2.run_isw_big_samples import copy_parameters


def save_models(epoch, rln, tln):
    try:
        isdir("saved_models/")
    except NotADirectoryError:
        makedirs("save_models")
    rln.save(f"saved_models/rln_pretraining_{epoch}.tf", save_format="tf")
    tln.save(f"saved_models/tln_pretraining_{epoch}.tf", save_format="tf")


@tf.function
def inner_update(x, y, tln, beta):
    with tf.GradientTape(watch_accessed_variables=False) as Wj_Tape:
        Wj_Tape.watch(tln.trainable_variables)
        inner_loss = compute_loss(x, y)
    gradients = Wj_Tape.gradient(inner_loss, tln.trainable_variables)
    for g, v in zip(gradients, tln.trainable_variables):
        v.assign_sub(beta * g)


@tf.function
def compute_loss(x, y, tln, rln, loss_fun):
    output = tln(rln(x))
    loss = loss_fun(output, y)
    return loss


def pretrain_mrcl(x_traj, y_traj, x_rand, y_rand, tln, tln_initial, rln, meta_optimizer):
    # Random reinitialization of last layer
    # w = tln.layers[-1].weights[0]
    # new_w = tln.layers[-1].kernel_initializer(shape=w.shape)
    # tln.layers[-1].weights[0].assign(new_w)

    copy_parameters(tln, tln_initial)

    # Sample x_rand, y_rand from s_remember
    x_shape = x_traj.shape
    x_traj_f = tf.reshape(x_traj, (x_shape[0] * x_shape[1], x_shape[2]))
    y_traj_f = tf.reshape(y_traj, (x_shape[0] * x_shape[1],))

    x_meta = tf.concat([x_rand, x_traj_f], axis=0)
    y_meta = tf.concat([y_rand, y_traj_f], axis=0)

    for x, y in tf.data.Dataset.from_tensor_slices((x_traj, y_traj)):
        inner_update(x, y)

    with tf.GradientTape(persistent=True) as theta_Tape:
        outer_loss = compute_loss(x_meta, y_meta)

    tln_gradients = theta_Tape.gradient(outer_loss, tln.trainable_variables)
    rln_gradients = theta_Tape.gradient(outer_loss, rln.trainable_variables)
    del theta_Tape
    meta_optimizer.apply_gradients(zip(tln_gradients + rln_gradients, tln_initial.trainable_variables + rln.trainable_variables))

    copy_parameters(tln_initial, tln)

    return outer_loss