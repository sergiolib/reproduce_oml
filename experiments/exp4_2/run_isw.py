import datetime
from os import makedirs
from random import sample
from shutil import rmtree
import tensorflow as tf
import tqdm
import numpy as np

from datasets.synth_datasets import gen_sine_data, gen_tasks
from experiments.exp4_2.isw import mrcl_isw
from experiments.training import pretrain_mrcl, save_models

parameters = {
    "inner_learning_rate": 3e-3,  # beta
    "meta_learning_rate": 1e-4,  # alpha
    "epochs": 20000,  # number of epochs to pre train for
    "n_tasks": 400,  # number of tasks to pre train from
    "n_functions": 10,  # number of functions to sample per epoch
    "sample_length": 32,  # length of each sequence sampled
    "repetitions": 40  # number of
}

rln, tln = mrcl_isw(one_hot_depth=parameters["n_functions"])  # Create and initialize models

tasks = gen_tasks(parameters["n_functions"])  # Generate tasks parameters

# Create logs directories
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/isw/' + current_time + '/train'
try:
    rmtree('logs')
except FileNotFoundError:
    pass
makedirs(train_log_dir)

# Create file writer for Tensorboard (logdir = ./logs/isw)
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

# Initialize loss function and meta optimizer
loss_fun = tf.keras.losses.MeanSquaredError()
meta_optimizer = tf.keras.optimizers.Adam(learning_rate=parameters["meta_learning_rate"])

# Fabricate TLN clone for storing the parameters at the beginning of each iteration
tln_initial = tf.keras.models.clone_model(tln)

# Main pre training loop
for epoch in tqdm.trange(parameters["epochs"]):
    x_traj, y_traj, x_rand, y_rand = gen_sine_data(tasks=tasks, n_functions=parameters["n_functions"],
                                                   sample_length=parameters["sample_length"],
                                                   repetitions=parameters["repetitions"])

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

    # Pretrain step
    loss = pretrain_mrcl(x_traj=x_traj, y_traj=y_traj,
                         x_rand=x_rand, y_rand=y_rand,
                         rln=rln, tln=tln, tln_initial=tln_initial,
                         meta_optimizer=meta_optimizer,
                         loss_function=loss_fun,
                         beta=parameters["inner_learning_rate"])

    # Check metrics
    rep = rln(x_rand)
    rep = np.array(rep)
    counts = np.isclose(rep, 0).sum(axis=1) / rep.shape[1]
    sparsity = np.mean(counts)
    with train_summary_writer.as_default():
        tf.summary.scalar('Sparsity', sparsity, step=epoch)
        tf.summary.scalar('Training loss', loss, step=epoch)

    if epoch % 100 == 0 and epoch > 0:
        save_models(epoch)

    if epoch % 100 == 0:
        ix = sample(range(x_rand.shape[0]), 1)
        x = tf.reshape(x_rand[ix], (1, -1))
        rep = rln(x)
        rep = tf.reshape(rep[0], (30, 10))
        rep = rep / tf.reduce_max(rep)
        rep = tf.reshape(rep, (1, 30, 10, 1))
        tf.summary.image("representation", rep, epoch)
