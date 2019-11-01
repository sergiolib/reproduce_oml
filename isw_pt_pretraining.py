import argparse
import json
import math

import datetime
import os
import tqdm
import numpy as np
import tensorflow as tf

from datasets.synth_datasets import gen_sine_data, gen_tasks
from experiments.exp4_2.isw import mrcl_isw
from experiments.training import pretrain_mrcl, save_models, to_iid
from experiments.training import copy_parameters, prepare_data_pre_training
from experiments.evaluation import evaluate_models_isw, prepare_data_evaluation
from experiments.evaluation import compute_sparsity, get_representations_graphics
from baseline_methods.pretraining import PretrainingBaseline


model_prefix = "isw_basicpt"

def main(args):
    tr_tasks = gen_tasks(args.n_tasks)  # Generate tasks parameters
    val_tasks = gen_tasks(args.val_tasks)
    loss_fun = tf.keras.losses.MeanSquaredError()

    # Create logs directories
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = "/".join(["logs", model_prefix, current_time, "train"])
    os.makedirs(train_log_dir, exist_ok=True)

    # Main pre training loop
    # Create and initialize models
    pb = PretrainingBaseline(tf.keras.losses.MeanSquaredError())
    pb.build_isw_model()

    # Create file writer for Tensorboard (logdir = ./logs/isw)
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    # Fabricate TLN clone for storing the parameters at the beginning of each
    # iteration
    tln_copy = tf.keras.models.clone_model(pb.model_tln)

    # Validation and online training data
    val_data = prepare_data_evaluation(val_tasks,
                                       args.n_functions,
                                       args.sample_length,
                                       args.val_repetitions)
    x_train, y_train, x_val, y_val = val_data

    eval_optimizer = tf.keras.optimizers.SGD(learning_rate=0.003)

    for epoch in tqdm.trange(args.epochs):
        tr_data = prepare_data_pre_training(tr_tasks,
                                            args.n_functions,
                                            args.sample_length,
                                            args.pt_repetitions)
        x_traj, y_traj, _, _ = tr_data
        x_traj, y_traj = to_iid(x_traj, y_traj, args.n_functions,
args.sample_length, args.pt_repetitions)

        # Pretrain step
        pt_loss = pb.pre_train(x_traj, y_traj, args.learning_rate)

        # Check metrics for Tensorboard to be included every
        # "post_results_every" epochs
        if epoch % args.post_results_every == 0:
            sparsity = compute_sparsity(x_val, pb.model_rln, pb.model_tln)

            with train_summary_writer.as_default():
                tf.summary.scalar('Sparsity', sparsity, step=epoch)
                tf.summary.scalar('Training loss', pt_loss,
                                  step=epoch)

            rep = get_representations_graphics(x_val, pb.model_rln)
            with train_summary_writer.as_default():
                tf.summary.image("representation", rep, epoch)

            copy_parameters(pb.model_tln, tln_copy)

            losses = evaluate_models_isw(x_train=x_train,
                                         y_train=y_train,
                                         x_val=x_val,
                                         y_val=y_val,
                                         tln=tln_copy, rln=pb.model_rln)

            mean_loss_all_val = losses[0]
            # loss_per_class_during_training = losses[1]

            with train_summary_writer.as_default():
                tf.summary.scalar('Validation loss', mean_loss_all_val, step=epoch)
            print(f"Epoch: {epoch}\tSparsity: {sparsity}\t"
                  f"Mean loss: {mean_loss_all_val}")

        # Save model every "save_models_every" epochs
        if epoch % args.save_models_every == 0 and epoch > 0:
            save_models(model=rln, name=model_prefix + f"_rln")
            save_models(model=tln, name=model_prefix + f"_tln")

    # Save final model
    save_models(model=rln, name=model_prefix + f"_rln")
    save_models(model=tln, name=model_prefix + f"_tln")

if __name__ == '__main__':
    # Parse arguments
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--learning_rate", type=float, default=3e-3,
                                 help="Learning rate")
    argument_parser.add_argument("--epochs", type=int, default=20000,
                                 help="number of epochs to pre train for")
    argument_parser.add_argument("--n_tasks", type=int, default=400,
                                 help="number of tasks to pre train from")
    argument_parser.add_argument("--val_tasks", type=int, default=400,
                                 help="number of validation tasks to train and evaluate from")
    argument_parser.add_argument("--n_functions", type=int, default=10,
                                 help="number of functions to sample per epoch")
    argument_parser.add_argument("--sample_length", type=int, default=32,
                                 help="length of each sequence sampled")
    argument_parser.add_argument("--pt_repetitions", type=int, default=40,
                                 help="number of pre train repetitions for generating"
                                      " the data samples")
    argument_parser.add_argument("--val_repetitions", type=int, default=50,
                                 help="number of validation/train repetitions for generating"
                                      " the data samples")
    argument_parser.add_argument("--save_models_every", type=int, default=100,
                                 help="Amount of epochs to pass before saving"
                                      " models")
    argument_parser.add_argument("--post_results_every", type=int, default=200,
                                 help="Amount of epochs to pass before posting"
                                      " results in Tensorboard")
    argument_parser.add_argument("--resetting_last_layer", default=True, type=bool,
                                 help="Reinitialization of the last layer of"
                                      " the TLN")
    argument_parser.add_argument("--representation_size", default=900, type=int,
                                 help="Size of representations")

    args = argument_parser.parse_args()
    main(args)
