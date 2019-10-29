import datetime
import os
import tensorflow as tf
import tqdm
import numpy as np
import json

from datasets.synth_datasets import gen_sine_data, gen_tasks
from experiments.evaluation import train_and_evaluate

import argparse

# Arguments parsing
argument_parser = argparse.ArgumentParser()

argument_parser.add_argument("--n_tasks", default=500, type=int,
                             help="Number of tasks to pre train from")
argument_parser.add_argument("--n_functions", default=10, type=int,
                             help="Number of functions to sample per epoch")
argument_parser.add_argument("--sample_length", default=32, type=int,
                             help="Length of each sequence sampled")
argument_parser.add_argument("--repetitions", default=50, type=int,
                             help="Number of train repetitions for generating the data samples")
argument_parser.add_argument("--batch_size_evaluation", default=8, type=int,
                             help="Batch size for evaluation stage training")
argument_parser.add_argument("--tests", default=50, type=int,
                             help="Times to test and get results / number of random trajectories")
argument_parser.add_argument("--model_file_rln", default="saved_models/pt_lr3e-06_rln6_tln2_rln.tf", type=str,
                             help="Model file for the rln")
argument_parser.add_argument("--model_file_tln", default="saved_models/pt_lr3e-06_rln6_tln2_tln.tf", type=str,
                             help="Model file for the tln")
argument_parser.add_argument("--learning_rate", nargs="+", default=0.000003,
                             type=float, help="Learning rate(s) to try")
argument_parser.add_argument("--results_dir", default="./results/basic_pt_isw/", type=str,
                             help="Evaluation results file")
argument_parser.add_argument("--resetting_last_layer", default=True, type=bool,
                             help="Reinitialization of the last layer of the TLN")

args = argument_parser.parse_args()

tasks = gen_tasks(args.n_functions)  # Generate tasks parameters
test_tasks = gen_tasks(args.n_tasks)
loss_fun = tf.keras.losses.MeanSquaredError()

# Create logs directories
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/pt_isw/' + current_time + '/eval'
os.makedirs(train_log_dir, exist_ok=True)

_, _, x_val, y_val = gen_sine_data(test_tasks,
                                   args.n_functions,
                                   args.sample_length,
                                   args.repetitions)
x_val = tf.convert_to_tensor(x_val, dtype=tf.float32)
y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)
learning_rate = args.learning_rate if type(args.learning_rate) is list else [args.learning_rate]

# Find a good learning rate from the given ones to iterate many times on
final_losses = []
for lr in tqdm.tqdm(learning_rates):
    # Continual Regression Experiment (Figure 3)
    x_train, y_train, _, _ = gen_sine_data(tasks,
                                           args.n_functions,
                                           args.sample_length,
                                           args.repetitions)

    # Reshape for inputting to training method
    x_train = np.transpose(x_train, (1, 2, 0, 3))
    y_train = np.transpose(y_train, (1, 2, 0))
    x_train = np.reshape(x_train, (args.repetitions * args.sample_length, args.n_functions, -1))
    y_train = np.reshape(y_train, (args.repetitions * args.sample_length, args.n_functions))
    x_train = np.transpose(x_train, (1, 0, 2))
    y_train = np.transpose(y_train, (1, 0))
    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)

    # Numpy -> Tensorflow
    tf.keras.backend.clear_session()
    rln = tf.keras.models.load_model(args.model_file_rln)
    tln = tf.keras.models.load_model(args.model_file_tln)

    if args.resetting_last_layer:
        # Random reinitialization of last layer
        w = tln.layers[-1].weights[0]
        b = tln.layers[-1].weights[1]
        new_w = tf.keras.initializers.he_normal()(shape=w.shape)
        w.assign(new_w)
        new_b = tf.keras.initializers.zeros()(shape=b.shape)
        b.assign(new_b)

    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    results = train_and_evaluate(x_train=x_train, y_train=y_train,
                                 x_test=x_val, y_test=y_val,
                                 rln=rln, tln=tln, optimizer=optimizer,
                                 loss_function=loss_fun, batch_size=args.batch_size_evaluation)

    final_losses.append((lr, results[args.n_functions]))

best_lr, best_loss = sorted(final_losses, key=lambda x: x[1])[0]

# Run many times with best_lr
lr_results.append(results)
optimizer = tf.keras.optimizers.SGD(learning_rate=best_lr)
all_results = []
for i in tqdm.tqdm(args.tests):
    # Continual Regression Experiment (Figure 3)
    x_train, y_train, _, _ = gen_sine_data(tasks,
                                           args.n_functions,
                                           args.sample_length,
                                           args.repetitions)

    # Reshape for inputting to training method
    x_train = np.transpose(x_train, (1, 2, 0, 3))
    y_train = np.transpose(y_train, (1, 2, 0))
    x_train = np.reshape(x_train, (args.repetitions * args.sample_length, args.n_functions, -1))
    y_train = np.reshape(y_train, (args.repetitions * args.sample_length, args.n_functions))
    x_train = np.transpose(x_train, (1, 0, 2))
    y_train = np.transpose(y_train, (1, 0))
    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)

    # Numpy -> Tensorflow
    tf.keras.backend.clear_session()
    rln = tf.keras.models.load_model(args.model_file_rln)
    tln = tf.keras.models.load_model(args.model_file_tln)

    if args.resetting_last_layer:
        # Random reinitialization of last layer
        w = tln.layers[-1].weights[0]
        b = tln.layers[-1].weights[1]
        new_w = tf.keras.initializers.he_normal()(shape=w.shape)
        w.assign(new_w)
        new_b = tf.keras.initializers.zeros()(shape=b.shape)
        b.assign(new_b)

    results = train_and_evaluate(x_train=x_train, y_train=y_train,
                                 x_test=x_val, y_test=y_val,
                                 rln=rln, tln=tln, optimizer=optimizer,
                                 loss_function=loss_fun, batch_size=args.batch_size_evaluation)

    all_results.append((lr, results[args.n_functions]))

location = os.path.join(args.results_dir, f"basic_pt_eval_{lr}.json")
os.makedirs(os.path.dirname(location), exist_ok=True)
json.dump(results, open(location, "w"))
