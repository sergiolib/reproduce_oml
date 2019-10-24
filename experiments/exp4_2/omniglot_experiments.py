import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../../datasets"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from tf_datasets import load_omniglot
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
from omniglot_model import mrcl_omniglot 
from training import copy_parameters 
from operator import itemgetter
import datetime
import os

print(f"GPU is available: {tf.test.is_gpu_available()}")

dataset = "omniglot"

background_data, evaluation_data = load_omniglot(dataset, verbose=1)

def get_data_by_classes(background_data, evaluation_data):
    background_data = np.array(sorted(list(tfds.as_numpy(background_data)), key=itemgetter('label')))
    evaluation_data = np.array(sorted(list(tfds.as_numpy(evaluation_data)), key=itemgetter('label')))
    background_training_data = []
    evaluation_training_data = []
    evaluation_test_data = []

    evaluation_number_of_classes = int(evaluation_data.shape[0]/20)
    background_number_of_classes = int(background_data.shape[0]/20)
    for i in range(evaluation_number_of_classes):
        evaluation_training_data.append(evaluation_data[i*20:i*20+15])
        evaluation_test_data.append(evaluation_data[i*20+15:(i+1)*20])

    for i in range(background_number_of_classes):
        background_training_data.append(background_data[i*20:(i+1)*20])
    return background_training_data, evaluation_training_data, evaluation_test_data

def partition_into_disjoint(data):
    number_of_classes = len(data)
    s_learn = list(range(0, int(number_of_classes/2)))
    s_remember = list(range(int(number_of_classes/2), number_of_classes))
    return s_learn, s_remember

def sample_trajectory(s_learn, data):
    random_class = np.random.choice(s_learn)
    x_traj = []
    y_traj = []
    for item in data[random_class]:
        x_traj.append(item['image'])
        y_traj.append(item['label'])
    return tf.convert_to_tensor(x_traj), tf.convert_to_tensor(y_traj)

def sample_random(s_remember, data):
    random_class = np.random.choice(s_remember)
    x_rand = []
    y_rand = []
    for item in data[random_class]:
        x_rand.append(item['image'])
        y_rand.append(item['label'])
    return tf.convert_to_tensor(x_rand), tf.convert_to_tensor(y_rand)

def pretrain_classification_mrcl(x_traj, y_traj, x_rand, y_rand):
    # Random reinitialization of last layer
    w = tln.layers[-1].weights[0]
    new_w = tln.layers[-1].kernel_initializer(shape=w.shape)
    tln.layers[-1].weights[0].assign(new_w)

    # Clone tln to preserve initial weights
    tln_initial = tf.keras.models.clone_model(tln)

    x_meta = tf.concat([x_rand, x_traj], axis=0)
    y_meta = tf.concat([y_rand, y_traj], axis=0)

    for x, y in zip(x_traj, y_traj):
        inner_update(x, y)

    with tf.GradientTape(persistent=True) as theta_tape:
        outer_loss = compute_loss(x_meta, y_meta)

    tln_gradients = theta_tape.gradient(outer_loss, tln.trainable_variables)
    rln_gradients = theta_tape.gradient(outer_loss, rln.trainable_variables)
    del theta_tape
    meta_optimizer_outer.apply_gradients(zip(tln_gradients + rln_gradients, tln_initial.trainable_variables + rln.trainable_variables))

    copy_parameters(tln_initial, tln)

    return outer_loss

@tf.function
def inner_update(x, y):
    with tf.GradientTape(watch_accessed_variables=False) as Wj_Tape:
        Wj_Tape.watch(tln.trainable_variables)
        inner_loss = compute_loss(x, y)
    gradients = Wj_Tape.gradient(inner_loss, tln.trainable_variables)
    for g, v in zip(gradients, tln.trainable_variables):
        v.assign_sub(classification_parameters["inner_learning_rate"] * g)
    

@tf.function
def compute_loss(x, y):
    if (x.shape.ndims == 3):
        output = tln(rln(tf.expand_dims(x, axis=0)))
    else:
        output = tln(rln(x))
    loss = loss_fun(y, output)
    return loss

def save_models(rs):
    try:
        os.path.isdir(os.path.join(os.path.dirname(__file__), "saved_models"))
    except NotADirectoryError:
        print("Creating a directory")
        os.makedirs(os.path.join(os.path.dirname(__file__), "saved_models"))
    rln.save(os.path.join(os.path.dirname(__file__), f"saved_models/rln_pretraining_{rs}.h5"))
    tln.save(os.path.join(os.path.dirname(__file__), f"saved_models/tln_pretraining_{rs}.h5"))

background_training_data, evaluation_training_data, evaluation_test_data = get_data_by_classes(background_data, evaluation_data)

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
    "loss_metric": tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    "online_optimizer": tf.optimizers.SGD,
    "meta_optimizer": tf.optimizers.Adam
}

t = range(20000)
tasks = None

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/classification/gradient_tape/' + current_time + '/train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
loss_fun = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
meta_optimizer_inner = tf.keras.optimizers.SGD(learning_rate=classification_parameters["inner_learning_rate"])
meta_optimizer_outer = tf.keras.optimizers.Adam(learning_rate=classification_parameters["meta_learning_rate"])

for epoch, v in enumerate(t):
    x_rand, y_rand = sample_random(s_remember, background_training_data)
    x_traj, y_traj = sample_trajectory(s_learn, background_training_data)
    loss = pretrain_classification_mrcl(x_traj, y_traj, x_rand, y_rand)
    
    # Check metrics
    rep = rln(x_rand)
    rep = np.array(rep)
    counts = np.isclose(rep, 0).sum(axis=1) / rep.shape[1]
    sparsity = np.mean(counts)
    with train_summary_writer.as_default():
        tf.summary.scalar('Sparsity', sparsity, step=epoch)
        tf.summary.scalar('Training loss', loss, step=epoch)
    print("Epoch:", epoch, "Sparsity:", sparsity, "Training loss:", loss.numpy())