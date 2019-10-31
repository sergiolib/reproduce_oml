import json
import math

def factor_int(n):
    nsqrt = math.ceil(math.sqrt(n))
    solution = False
    val = nsqrt
    while not solution:
        val2 = int(n/val)
        if val2 * val == float(n):
            solution = True
        else:
            val-=1
    return val, val2


import argparse
# Parse arguments
argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("--meta_learning_rate", type=float, default=1e-4,
                             help="alpha")
argument_parser.add_argument("--inner_learning_rate", type=float, default=3e-3,
                             help="beta")
argument_parser.add_argument("--epochs", type=int, default=20000,
                             help="number of epochs to pre train for")
argument_parser.add_argument("--n_tasks", type=int, default=400,
                             help="number of tasks to pre train from")
argument_parser.add_argument("--n_functions", type=int, default=10,
                             help="number of functions to sample per epoch")
argument_parser.add_argument("--sample_length", type=int, default=32,
                             help="length of each sequence sampled")
argument_parser.add_argument("--repetitions", type=int, default=40,
                             help="number of train repetitions for generating"
                                  " the data samples")
argument_parser.add_argument("--save_models_every", type=int, default=100,
                             help="Amount of epochs to pass before saving"
                                  " models")
argument_parser.add_argument("--post_results_every", type=int, default=50,
                             help="Amount of epochs to pass before posting"
                                  " results in Tensorboard")
argument_parser.add_argument("--resetting_last_layer", default=True, type=bool,
                             help="Reinitialization of the last layer of"
                                  " the TLN")
argument_parser.add_argument("--representation_size", default=900, type=int,
                             help="Size of representations")

args = argument_parser.parse_args()

import datetime
import os

import numpy as np
import tensorflow as tf

from datasets.synth_datasets import gen_sine_data, gen_tasks
from experiments.exp4_2.isw import mrcl_isw
from experiments.training import pretrain_mrcl, save_models
from experiments.evaluation import train_and_evaluate

from experiments.training import copy_parameters

tasks = gen_tasks(args.n_tasks)  # Generate tasks parameters
val_tasks = gen_tasks(500)
loss_fun = tf.keras.losses.MeanSquaredError()

# Create logs directories
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/isw/' + current_time + '/train'
os.makedirs(train_log_dir, exist_ok=True)

# Main pre training loop
# Create and initialize models
rln, tln = mrcl_isw(one_hot_depth=args.n_functions, representation_size=args.representation_size)

# Create file writer for Tensorboard (logdir = ./logs/isw)
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

# Initialize loss function and meta optimizer
mlr = args.meta_learning_rate
meta_optimizer = tf.keras.optimizers.Adam(learning_rate=mlr)

# Fabricate TLN clone for storing the parameters at the beginning of each
# iteration
tln_copy = tf.keras.models.clone_model(tln)

_, _, x_val, y_val = gen_sine_data(tasks=val_tasks,
                                   n_functions=args.n_functions,
                                   sample_length=args.sample_length,
                                   repetitions=args.repetitions)

# Reshape for inputting to training method
x_val = np.vstack(x_val)
y_val = np.hstack(y_val)

# Numpy -> Tensorflow
x_val = tf.convert_to_tensor(x_val, dtype=tf.float32)
y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)

n_functions = args.n_functions
sl = args.sample_length
reps = args.repetitions

eval_optimizer = tf.keras.optimizers.SGD(learning_rate=0.003)

for epoch in range(args.epochs):
    # Sample data
    x_traj, y_traj, x_rand, y_rand = gen_sine_data(tasks=tasks,
                                                   n_functions=n_functions,
                                                   sample_length=sl,
                                                   repetitions=reps)

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
                         rln=rln, tln=tln, tln_initial=tln_copy,
                         meta_optimizer=meta_optimizer,
                         loss_function=loss_fun,
                         beta=args.inner_learning_rate,
                         reset_last_layer=args.resetting_last_layer)

    # Check metrics for Tensorboard to be included every "post_results_every"
    # epochs
    if epoch % args.post_results_every == 0:
        rep = rln(x_rand)
        rep = np.array(rep)
        counts = np.isclose(rep, 0).sum(axis=1) / rep.shape[1]
        sparsity = np.mean(counts)
        with train_summary_writer.as_default():
            tf.summary.scalar('Sparsity', sparsity, step=epoch)
            tf.summary.scalar('Training loss', loss, step=epoch)
        print(f"Epoch: {epoch}\tSparsity: {sparsity}\tTraining loss: {loss}")

    # Save model every "save_models_every" epochs
    if epoch % args.save_models_every == 0 and epoch > 0:
        save_models(model=rln, name=f"isw_mrcl_rln")
        save_models(model=tln, name=f"isw_mrcl_tln")

    if epoch % args.post_results_every == 0:
        x = x_val
        rep = rln(x)
        rep_len = rep.shape[-1]
        rep_f1, rep_f2 = factor_int(rep_len)
        rep = [tf.reshape(r, (rep_f1, rep_f2, 1)) for r in rep]
        rep = [r / tf.reduce_max(r) for r in rep]
        rep = tf.random.shuffle(tf.stack(rep))
        with train_summary_writer.as_default():
            tf.summary.image("representation", rep, epoch)

    if epoch % args.post_results_every == 0:
        x_train, y_train, x_val, y_val = gen_sine_data(val_tasks,
                                                       args.n_functions,
                                                       args.sample_length,
                                                       args.repetitions)

        # Reshape for inputting to training method
        x_train = np.transpose(x_train, (1, 2, 0, 3))
        y_train = np.transpose(y_train, (1, 2, 0))
        x_train = np.reshape(x_train, (args.repetitions * args.sample_length,
                                       args.n_functions, -1))
        y_train = np.reshape(y_train, (args.repetitions * args.sample_length,
                                       args.n_functions))
        x_train = np.transpose(x_train, (1, 0, 2))
        y_train = np.transpose(y_train, (1, 0))

        # Numpy -> Tensorflow
        x_val = tf.convert_to_tensor(x_val, dtype=tf.float32)
        y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)
        x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
        y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)

        copy_parameters(tln, tln_copy)

        if args.resetting_last_layer:
            # Random reinitialization of last layer
            w = tln_copy.layers[-1].weights[0]
            b = tln_copy.layers[-1].weights[1]
            new_w = tf.keras.initializers.he_normal()(shape=w.shape)
            w.assign(new_w)
            new_b = tf.keras.initializers.zeros()(shape=b.shape)
            b.assign(new_b)
        results = train_and_evaluate(x_train=x_train, y_train=y_train,
                                     x_val=x_val, y_val=y_val, rln=rln,
                                     tln=tln_copy, optimizer=eval_optimizer,
                                     loss_function=loss_fun,
                                     batch_size=8)
        loss_classes_seen, interference = results

        json.dump(loss_classes_seen, "./results/isw_mrcl/isw_mrcl_eval_0.003_3_pretraining_0.003_{epoch}_3a.json")
        json.dump(interference, "./results/isw_mrcl/isw_mrcl_eval_0.003_3_pretraining_0.003_{epoch}_3b.json")


# Save final model
save_models(model=rln, name=f"isw_mrcl_rln")
save_models(model=tln, name=f"isw_mrcl_tln")
