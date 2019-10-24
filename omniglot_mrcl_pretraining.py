import tensorflow as tf
import datetime
import numpy as np
from experiments.exp4_2.omniglot_model import mrcl_omniglot, get_data_by_classes, \
    partition_into_disjoint, pretrain_classification_mrcl, sample_trajectory, sample_random, sample_random_10_classes
from datasets.tf_datasets import load_omniglot
from experiments.training import save_models

print(f"GPU is available: {tf.test.is_gpu_available()}")

dataset = "omniglot"
background_data, evaluation_data = load_omniglot(dataset, verbose=1)
background_training_data, evaluation_training_data, evaluation_test_data = get_data_by_classes(background_data,
                                                                                               evaluation_data)
assert len(evaluation_training_data) == 659
assert len(evaluation_test_data) == 659
assert len(background_training_data) == 964

assert evaluation_training_data[0].shape[0] == 15
assert evaluation_test_data[0].shape[0] == 5
assert background_training_data[0].shape[0] == 20

s_learn, s_remember = partition_into_disjoint(background_training_data)

rln, tln = mrcl_omniglot()
print(rln.summary())
print(tln.summary())

classification_parameters = {
    "meta_learning_rate": 1e-4,
    "inner_learning_rate": 0.03,
    "loss_function": tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    "online_optimizer": tf.optimizers.SGD,
    "meta_optimizer": tf.optimizers.Adam
}

t = range(20000)
tasks = None

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/classification/gradient_tape/' + current_time + '/train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

for epoch, v in enumerate(t):
    x_rand, y_rand = sample_random_10_classes(s_remember, background_training_data)
    x_traj, y_traj = sample_trajectory(s_learn, background_training_data)
    loss = pretrain_classification_mrcl(x_traj, y_traj, x_rand, y_rand, rln, tln, classification_parameters)

    # Check metrics
    rep = rln(x_rand)
    rep = np.array(rep)
    counts = np.isclose(rep, 0).sum(axis=1) / rep.shape[1]
    sparsity = np.mean(counts)
    with train_summary_writer.as_default():
        tf.summary.scalar('Sparsity', sparsity, step=epoch)
        tf.summary.scalar('Training loss', loss, step=epoch)
    print("Epoch:", epoch, "Sparsity:", sparsity, "Training loss:", loss.numpy())
    if epoch % 100 == 0:
        save_models(tln, f"tln_pretraining_mrcl_{epoch}_omniglot")
        save_models(tln, f"rln_pretraining_mrcl_{epoch}_omniglot")