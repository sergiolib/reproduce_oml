import argparse
import datetime
from itertools import product
from os import makedirs

import numpy as np
import tensorflow as tf

from experiments.exp4_2.omniglot_model import get_data_by_classes
from datasets.tf_datasets import load_omniglot
from baseline_methods.pretraining import PretrainingBaseline

gpus = tf.config.experimental.list_physical_devices('GPU')
# if len(gpus) > 0:
#     tf.config.experimental.set_virtual_device_configuration(gpus[0], [
#         tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])

# Parse arguments
argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("--learning_rate", nargs="+", type=float, default=[3e-6],
                             help="Learning rate")
argument_parser.add_argument("--epochs", type=int, default=5000,
                             help="Number of epochs to pre train for")
# argument_parser.add_argument("--n_tasks", type=int, default=400,
#                              help="Number of tasks to pre train from")
# argument_parser.add_argument("--n_functions", type=int, default=10,
#                              help="Number of functions to sample per epoch")
# argument_parser.add_argument("--sample_length", type=int, default=32,
#                              help="Length of each sequence sampled")
argument_parser.add_argument("--repetitions", type=int, default=40,
                             help="Number of train repetitions for generating the data samples")
argument_parser.add_argument("--save_models_every", type=int, default=100,
                             help="Amount of epochs to pass before saving models")
argument_parser.add_argument("--check_val_every", type=int, default=100,
                             help="Amount of epochs to pass before checking validation loss")
argument_parser.add_argument("--n_layers_rln", type=int, nargs="+", default=5,
                             help="Amount of layers in the RLN")
argument_parser.add_argument("--n_layers_tln", type=int, nargs="+", default=5,
                             help="Amount of layers in the TLN")

args = argument_parser.parse_args()

# Generate dataset

dataset = "omniglot"
background_data, evaluation_data = load_omniglot(dataset, verbose=1)
_, evaluation_training_data, evaluation_test_data = get_data_by_classes(background_data, evaluation_data)
all_classes = list(range(len(evaluation_training_data)))
classes_to_use = np.random.choice(all_classes, 200)
seen_classes = 0

x_val = []
y_val = []
temp_class_id = 0
for class_id in classes_to_use:
    for testing_item in evaluation_training_data[class_id]:
        x_val.append(testing_item['image'])
        y_val.append(temp_class_id)
    temp_class_id = temp_class_id + 1

x_val = tf.convert_to_tensor(x_val)
y_val = tf.convert_to_tensor(y_val)

# Main pre training loop
loss = float("inf")
# Create logs directories
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
n_layers_tln = [args.n_layers_tln] if type(args.n_layers_tln) is int else args.n_layers_tln
n_layers_rln = [args.n_layers_rln] if type(args.n_layers_rln) is int else args.n_layers_rln
gen = product(n_layers_tln, n_layers_rln, args.learning_rate)
p = PretrainingBaseline(tf.losses.SparseCategoricalCrossentropy(from_logits=True))

for tln_layers, rln_layers, lr in gen:
    print(f"tln: {tln_layers}, rln: {rln_layers}, lr: {lr}")
    train_log_dir = f'logs/pt_isw_lr{lr}_rln{rln_layers}_tln{tln_layers}/' + current_time + '/pre_train'
    makedirs(train_log_dir, exist_ok=True)
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    p.build_omniglot_model(n_layers_rln=rln_layers, n_layers_tln=tln_layers, seed=0)  # TODO: What architecture?
    val_loss_counts = 0
    previous_val_loss = p.compute_loss_no_training(x_val, y_val)
    val_loss = None

    for epoch in range(args.epochs):

        x_train = []
        y_train = []
        temp_class_id = 0
        for class_id in classes_to_use:
            for training_item in evaluation_training_data[class_id]:
                x_train.append(training_item['image'])
                y_train.append(temp_class_id)
            temp_class_id = temp_class_id + 1

        x_train = tf.convert_to_tensor(x_train)
        y_train = tf.convert_to_tensor(y_train)

        training_loss = float(p.pre_train(x_train, y_train, learning_rate=lr))

        with train_summary_writer.as_default():
            tf.summary.scalar("Training Loss", training_loss, step=epoch)

        if epoch % args.check_val_every == 0:
            val_loss = float(p.compute_loss_no_training(x_val, y_val))
            with train_summary_writer.as_default():
                tf.summary.scalar("Validation Loss", val_loss, step=epoch)

            if previous_val_loss - val_loss < 1e-3:
                val_loss_counts += 1
                if val_loss_counts == 1:
                    p.save_model(f"final_lr{lr}_rln{rln_layers}_tln{tln_layers}")
                elif val_loss_counts >= 6:
                    break
            else:
                previous_val_loss = float(val_loss)
                val_loss_counts = 0

        # if epoch % args.save_models_every == 0:
        #     p.save_model(f"{epoch}_lr{lr}_rln{rln_layers}_tln{tln_layers}")
        #
