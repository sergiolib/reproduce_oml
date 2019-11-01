import argparse

import datetime
import os

import tensorflow as tf

from datasets.synth_datasets import gen_tasks
from experiments.exp4_2.isw import mrcl_isw
from experiments.training import pretrain_mrcl, save_models
from experiments.training import copy_parameters, prepare_data_pre_training
from experiments.evaluation import evaluate_models_isw, prepare_data_evaluation
from experiments.evaluation import compute_sparsity
from experiments.evaluation import get_representations_graphics


model_prefix = "isw_mrcl"


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
    rln, tln = mrcl_isw(one_hot_depth=args.n_functions,
                        representation_size=args.representation_size)

    # Create file writer for Tensorboard (logdir = ./logs/isw)
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    # Initialize loss function and meta optimizer
    mlr = args.meta_learning_rate
    meta_optimizer = tf.keras.optimizers.Adam(learning_rate=mlr)

    # Fabricate TLN clone for storing the parameters at the beginning of each
    # iteration
    tln_copy = tf.keras.models.clone_model(tln)

    # Validation and online training data
    val_data = prepare_data_evaluation(val_tasks,
                                       args.n_functions,
                                       args.sample_length,
                                       args.val_repetitions)
    x_train, y_train, x_val, y_val = val_data

    eval_optimizer = tf.keras.optimizers.SGD(learning_rate=0.003)

    for epoch in range(args.epochs):
        tr_data = prepare_data_pre_training(tr_tasks,
                                            args.n_functions,
                                            args.sample_length,
                                            args.pt_repetitions)
        x_traj, y_traj, x_rand, y_rand = tr_data

        # Pretrain step
        pt_loss = pretrain_mrcl(x_traj=x_traj, y_traj=y_traj,
                                x_rand=x_rand, y_rand=y_rand,
                                rln=rln, tln=tln, tln_initial=tln_copy,
                                meta_optimizer=meta_optimizer,
                                loss_function=loss_fun,
                                beta=args.inner_learning_rate / 10,
                                reset_last_layer=args.resetting_last_layer)
        # Check metrics for Tensorboard to be included every
        # "post_results_every" epochs
        if epoch % args.post_results_every == 0:
            sparsity = compute_sparsity(x_rand, rln, tln)

            with train_summary_writer.as_default():
                tf.summary.scalar('Sparsity', sparsity, step=epoch)
                tf.summary.scalar('Training loss', pt_loss, step=epoch)

            rep = get_representations_graphics(x_val, rln)
            with train_summary_writer.as_default():
                tf.summary.image("representation", rep, epoch)

            copy_parameters(tln, tln_copy)

            losses = evaluate_models_isw(x_train=x_train,
                                         y_train=y_train,
                                         x_val=x_val,
                                         y_val=y_val,
                                         tln=tln_copy, rln=rln)

            mean_loss_all_val = losses[0]
            # loss_per_class_during_training = losses[1]

            with train_summary_writer.as_default():
                tf.summary.scalar('Validation loss', mean_loss_all_val,
                                  step=epoch)
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
    argument_parser.add_argument("--meta_learning_rate", type=float,
                                 default=1e-4,
                                 help="alpha")
    argument_parser.add_argument("--inner_learning_rate", type=float,
                                 default=3e-3, help="beta")
    argument_parser.add_argument("--epochs", type=int, default=20000,
                                 help="number of epochs to pre train for")
    argument_parser.add_argument("--n_tasks", type=int, default=400,
                                 help="number of tasks to pre train from")
    argument_parser.add_argument("--val_tasks", type=int, default=400,
                                 help="number of validation tasks to train and"
                                      " evaluate from")
    argument_parser.add_argument("--n_functions", type=int, default=10,
                                 help="number of functions to sample per epoch")
    argument_parser.add_argument("--sample_length", type=int, default=32,
                                 help="length of each sequence sampled")
    argument_parser.add_argument("--pt_repetitions", type=int, default=40,
                                 help="number of pre train repetitions for"
                                      " generating the data samples")
    argument_parser.add_argument("--val_repetitions", type=int, default=50,
                                 help="number of validation/train repetitions"
                                      " for generating the data samples")
    argument_parser.add_argument("--save_models_every", type=int, default=100,
                                 help="Amount of epochs to pass before saving"
                                      " models")
    argument_parser.add_argument("--post_results_every", type=int, default=200,
                                 help="Amount of epochs to pass before posting"
                                      " results in Tensorboard")
    argument_parser.add_argument("--resetting_last_layer", default=True,
                                 type=bool, help="Reinitialization of the last"
                                                 " layer of the TLN")
    argument_parser.add_argument("--representation_size", default=900,
                                 type=int, help="Size of representations")

    args = argument_parser.parse_args()
    main(args)
