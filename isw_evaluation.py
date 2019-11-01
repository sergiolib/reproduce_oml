import datetime
import os
import tensorflow as tf
import tqdm
import numpy as np
import json

from datasets.synth_datasets import gen_sine_data, gen_tasks
from experiments.evaluation import train_and_evaluate, prepare_data_evaluation
from experiments.evaluation import evaluate_models_isw

import argparse


def parse_args():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("model_name", type=str,
                                 help="Model name")
    argument_parser.add_argument("--n_tasks", default=500, type=int,
                                 help="Number of tasks to pre train from")
    argument_parser.add_argument("--n_functions", default=10, type=int,
                                 help="Number of functions to sample per epoch")
    argument_parser.add_argument("--sample_length", default=32, type=int,
                                 help="Length of each sequence sampled")
    argument_parser.add_argument("--repetitions", default=50, type=int,
                                 help="Number of train repetitions for"
                                      "generating the data samples")
    argument_parser.add_argument("--batch_size_evaluation", default=8, type=int,
                                 help="Batch size for evaluation stage training")
    argument_parser.add_argument("--tests", default=50, type=int,
                                 help="Times to test and get results / "
                                      "number of random trajectories")
    argument_parser.add_argument("--model_file_rln", default="saved_models/"
                                 "pt_lr1e-07_rln5_tln3_rln.tf", type=str,
                                 help="Model file for the rln")
    argument_parser.add_argument("--model_file_tln", default="saved_models/"
                                 "pt_lr1e-07_rln5_tln3_tln.tf", type=str,
                                 help="Model file for the tln")
    argument_parser.add_argument("--learning_rate", nargs="+",
                                 default=0.000003, type=float,
                                 help="Learning rate(s) to try")
    argument_parser.add_argument("--results_dir",
                                 default="./results/basic_pt_isw/", type=str,
                                 help="Evaluation results file")
    argument_parser.add_argument("--resetting_last_layer", default=True,
                                 type=bool, help="Reinitialization of the last"
                                                 " layer of the TLN")
    argument_parser.add_argument("--seed", default=0, type=int,
                                 help="Seed for the random functions")

    args = argument_parser.parse_args()
    return args

def parse_args_dbg():
    argument_parser = argparse.ArgumentParser()
    # argument_parser.add_argument("model_name", type=str,
    #                              help="Model name")
    argument_parser.add_argument("--n_tasks", default=500, type=int,
                                 help="Number of tasks to pre train from")
    argument_parser.add_argument("--n_functions", default=10, type=int,
                                 help="Number of functions to sample per epoch")
    argument_parser.add_argument("--sample_length", default=32, type=int,
                                 help="Length of each sequence sampled")
    argument_parser.add_argument("--repetitions", default=50, type=int,
                                 help="Number of train repetitions for"
                                      "generating the data samples")
    argument_parser.add_argument("--batch_size_evaluation", default=8, type=int,
                                 help="Batch size for evaluation stage training")
    argument_parser.add_argument("--tests", default=50, type=int,
                                 help="Times to test and get results / "
                                      "number of random trajectories")
    argument_parser.add_argument("--model_file_rln", default="/home/sliberman/Desktop/saved_models/isw_basicpt_rln.tf"
                                 "", type=str,
                                 help="Model file for the rln")
    argument_parser.add_argument("--model_file_tln", default="/home/sliberman/Desktop/saved_models/isw_basicpt_rln.tf", type=str,
                                 help="Model file for the tln")
    argument_parser.add_argument("--learning_rate", nargs="+",
                                 default=0.000003, type=float,
                                 help="Learning rate(s) to try")
    argument_parser.add_argument("--results_dir",
                                 default="./results/basic_pt_isw/", type=str,
                                 help="Evaluation results file")
    argument_parser.add_argument("--resetting_last_layer", default=True,
                                 type=bool, help="Reinitialization of the last"
                                                 " layer of the TLN")
    argument_parser.add_argument("--seed", default=0, type=int,
                                 help="Seed for the random functions")

    args = argument_parser.parse_args()
    args.model_name = "sergio"
    return args

def main(args):
    # Generate tasks parameters
    tasks = gen_tasks(args.n_functions)
    test_tasks = gen_tasks(args.n_tasks)

    # Loss function
    loss_fun = tf.keras.losses.MeanSquaredError()

    # Create logs directories
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/pt_isw/' + current_time + '/eval'
    os.makedirs(train_log_dir, exist_ok=True)

    # Find a good learning rate from the given ones to iterate many times on
    loss_per_lr = []
    all_mean_losses = []
    all_3a_results = []
    all_3b_results = []

    # Continual Regression Experiment (Figure 3)
    if type(args.learning_rate) is list:
        learning_rate = args.learning_rate
    else:
        learning_rate = [args.learning_rate]

    # Reshape for inputting to training method
    x_train, y_train, x_val, y_val = prepare_data_evaluation(tasks,
                                                             args.n_functions,
                                                             args.sample_length,
                                                             args.repetitions,
                                                             seed=args.seed)

    for lr in tqdm.tqdm(learning_rate):
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
        losses = evaluate_models_isw(x_train=x_train,
                                     y_train=y_train,
                                     x_val=x_val,
                                     y_val=y_val,
                                     tln=tln, rln=rln)

        mean_loss_all_val = losses[0]

        loss_per_lr.append((lr, mean_loss_all_val))

    best_lr, best_loss = sorted(loss_per_lr, key=lambda x: x[1])[0]

    # Run many times with best_lr
    optimizer = tf.keras.optimizers.SGD(learning_rate=best_lr)

    all_results = []
    for i in tqdm.trange(args.tests):
        # Continual Regression Experiment (Figure 3)
        data = prepare_data_evaluation(tasks,
                                       args.n_functions,
                                       args.sample_length,
                                       args.repetitions)

        x_train, y_train, x_val, y_val = data

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

        losses = evaluate_models_isw(x_train=x_train,
                                     y_train=y_train,
                                     x_val=x_val,
                                     y_val=y_val,
                                     rln=rln,
                                     tln=tln,
                                     reset_last_layer=True,
                                     batch_size=args.batch_size_evaluation,
                                     epochs=1)

        mean_loss_all_val = losses[0]
        loss_per_class_during_training = losses[1]
        interference_losses = losses[2]

        all_mean_losses.append(mean_loss_all_val)
        all_3a_results.append(loss_per_class_during_training)
        all_3b_results.append(interference_losses)

    location = os.path.join(args.results_dir, f"isw_{args.model_name}"
                                              f"losses_{lr}.json")
    os.makedirs(os.path.dirname(location), exist_ok=True)
    json.dump(all_mean_losses, open(location, "w"))
    location = os.path.join(args.results_dir, f"isw_{args.model_name}"
                                              f"3a_{lr}.json")
    json.dump(all_3a_results, open(location, "w"))
    location = os.path.join(args.results_dir, f"isw_{args.model_name}"
                                              f"3b_{lr}.json")
    json.dump(all_3b_results, open(location, "w"))

if __name__ == '__main__':
    args = parse_args()
    #args = parse_args_dbg()
    main(args)
