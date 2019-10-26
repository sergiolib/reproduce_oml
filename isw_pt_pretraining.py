import argparse
import datetime
from itertools import product
from os import makedirs

import numpy as np
import tensorflow as tf

from baseline_methods.pretraining import PretrainingBaseline
from datasets.synth_datasets import gen_sine_data, gen_tasks

gpus = tf.config.experimental.list_physical_devices('GPU')
f = tf.config.experimental.set_virtual_device_configurations
f(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])

# Parse arguments
argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("--learning_rate", nargs="+", default=[3e-3],
                             help="Learning rate")
argument_parser.add_argument("--epochs", type=int, default=20000,
                             help="Number of epochs to pre train for")
argument_parser.add_argument("--n_tasks", type=int, default=400,
                             help="Number of tasks to pre train from")
argument_parser.add_argument("--n_functions", type=int, default=10,
                             help="Number of functions to sample per epoch")
argument_parser.add_argument("--sample_length", type=int, default=32,
                             help="Length of each sequence sampled")
argument_parser.add_argument("--repetitions", type=int, default=40,
                             help="Number of train repetitions for generating the data samples")
argument_parser.add_argument("--save_models_every", type=int, default=100,
                             help="Amount of epochs to pass before saving models")
argument_parser.add_argument("--check_val_every", type=int, default=1,
                             help="Amount of epochs to pass before checking validation loss")

args = argument_parser.parse_args()

train_tasks = gen_tasks(args.n_functions)  # Generate tasks parameters
val_tasks = gen_tasks(10)

_, _, x_val, y_val = gen_sine_data(tasks=val_tasks, n_functions=args.n_functions,
                                   sample_length=args.sample_length,
                                   repetitions=1)

# Numpy -> Tensorflow
x_val = tf.convert_to_tensor(x_val, dtype=tf.float32)
y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)

# Main pre training loop
p = PretrainingBaseline(tf.keras.losses.MeanSquaredError())
loss = float("inf")
pre_train_not_finished = True
# Create logs directories
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
gen = product(list(range(1, 8)), args.learning_rate)
for rln_layers, lr in gen:
    tln_layers = 8 - rln_layers
    train_log_dir = f'logs/pt_isw_{lr}_rln{rln_layers}_tln{tln_layers}/' + current_time + '/train'
    makedirs(train_log_dir, exist_ok=True)
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    p.build_model(n_layers_rln=rln_layers, n_layers_tln=tln_layers)

    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    val_loss_counts = 0
    previous_val_loss = float("inf")
    val_loss = float("inf")
    epoch = 0
    while pre_train_not_finished:
        # Sample data
        x_train, y_train, _, _ = gen_sine_data(tasks=train_tasks, n_functions=args.n_functions,
                                               sample_length=args.sample_length,
                                               repetitions=args.repetitions)

        # Reshape for inputting to training method
        x_train = np.vstack(x_train)
        y_train = np.vstack(y_train)

        # Numpy -> Tensorflow
        x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
        y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
        x_train = tf.reshape(x_train, (-1, args.n_functions + 1))
        y_train = tf.reshape(y_train, (-1,))

        p.pre_train(x_train, y_train, optimizer, batch_size=8)

        previous_val_loss = val_loss

        if epoch % args.check_val_every == 0:
            val_loss = p.compute_loss(x_val, y_val)

            tf.summary.scalar("Validation Loss", val_loss, step=epoch)

            if previous_val_loss < val_loss:
                val_loss_counts += 1
                if val_loss_counts == 1:
                    p.save_model(f"final_lr{lr}_rln{rln_layers}_rln{tln_layers}")
                elif val_loss_counts >= 6:
                    break
            else:
                val_loss_counts = 0

        if epoch % args.save_models_every == 0:
            p.save_model(f"{epoch}_lr{lr}_rln{rln_layers}_tln{tln_layers}")

        epoch += 1
