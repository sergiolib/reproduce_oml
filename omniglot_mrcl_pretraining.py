import tensorflow as tf
import datetime
import numpy as np

from experiments.exp4_2.omniglot_model import mrcl_omniglot, get_background_data_by_classes, \
    partition_into_disjoint, pretrain_classification_mrcl, sample_trajectory, sample_random, sample_random_10_classes
from datasets.tf_datasets import load_omniglot
from experiments.training import save_models
from parameters import classification_parameters


def pretrain(sort_samples=True, model_name="mrcl"):
    print(f"GPU is available: {tf.test.is_gpu_available()}")

    background_data, _ = load_omniglot(verbose=1)
    background_training_data, _, _ = get_background_data_by_classes(background_data, sort=sort_samples)
    s_learn, s_remember = partition_into_disjoint(background_training_data)

    rln, tln = mrcl_omniglot()
    print(rln.summary())
    print(tln.summary())

    t = range(15000)
    tasks = None

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/classification/pretraining/omniglot/mrcl/gradient_tape/' + current_time + '/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    tln_initial = tf.keras.models.clone_model(tln)

    for epoch, v in enumerate(t):
        x_rand, y_rand = sample_random_10_classes(s_remember, background_training_data)
        x_traj, y_traj = sample_trajectory(s_learn, background_training_data)

        loss = pretrain_classification_mrcl(x_traj, y_traj, x_rand, y_rand, rln, tln, tln_initial, classification_parameters)

        # Check metrics
        rep = rln(x_rand)
        rep = np.array(rep)
        counts = np.isclose(rep, 0).sum(axis=1) / rep.shape[1]
        sparsity = np.mean(counts)
        with train_summary_writer.as_default():
            tf.summary.scalar('Sparsity', sparsity, step=epoch)
            tf.summary.scalar('Training loss', loss, step=epoch)
        print("Epoch:", epoch, "Sparsity:", sparsity, "Training loss:", loss.numpy())
        if epoch % 999 == 0:
            save_models(tln, f"tln_pretraining_{model_name}_{epoch}_omniglot")
            save_models(rln, f"rln_pretraining_{model_name}_{epoch}_omniglot")
