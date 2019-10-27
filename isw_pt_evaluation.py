import datetime
from os import makedirs
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
argument_parser.add_argument("--model_file_rln", default="saved_models/final_model_rln.tf", type=str,
                             help="Model file for the rln")
argument_parser.add_argument("--model_file_tln", default="saved_models/final_model_tln.tf", type=str,
                             help="Model file for the tln")
argument_parser.add_argument("--learning_rates", nargs="+",
                             default=[0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001, 0.000003, 0.000001],
                             type=float, help="Model file for the tln")
argument_parser.add_argument("--results_file", default="evaluation_results.json", type=str,
                             help="Evaluation results file")
argument_parser.add_argument("--resetting_last_layer", default=False, type=bool,
                             help="Reinitialization of the last layer of the TLN")

args = argument_parser.parse_args()

tasks = gen_tasks(args.n_functions)  # Generate tasks parameters
loss_fun = tf.keras.losses.MeanSquaredError()

# Create logs directories
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/pt_isw/' + current_time + '/eval'
makedirs(train_log_dir, exist_ok=True)

all_results = {}
for trajectory in tqdm.trange(args.tests):
    all_results[trajectory] = {}

    test_tasks = gen_tasks(args.n_tasks)
    x_train, y_train, x_val, y_val = gen_sine_data(test_tasks,
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

    # Numpy -> Tensorflow
    x_val = tf.convert_to_tensor(x_val, dtype=tf.float32)
    y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)
    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)

    # Continual Regression Experiment (Figure 3)
    for lr in tqdm.tqdm(args.learning_rates, leave=False):
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
        all_results[trajectory][lr] = results


json.dump(all_results, open(args.results_file, "w"))
