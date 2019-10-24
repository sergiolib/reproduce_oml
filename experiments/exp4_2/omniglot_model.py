from operator import itemgetter
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
from experiments.training import copy_parameters


def mrcl_omniglot_rln(inputs, n_layers=6, filters=256, strides=[2, 1, 2, 1, 2, 2]):
    h = inputs
    for i in range(n_layers):
        h = tf.keras.layers.Conv2D(filters, (3, 3), activation='relu', input_shape=(84, 84, 1), strides=strides[i])(h)
    h = tf.keras.layers.Flatten()(h)
    return h


def mrcl_omniglot_tln(inputs, n_layers=2, hidden_units_per_layer=300, output=964):
    y = inputs
    for i in range(n_layers - 1):
        y = tf.keras.layers.Dense(hidden_units_per_layer, activation='relu')(y)
    y = tf.keras.layers.Dense(output)(y)
    return y


def mrcl_omniglot(classes=964):
    input_rln = tf.keras.Input(shape=(84, 84, 1))
    input_tln = tf.keras.Input(shape=3 * 3 * 256)
    h = mrcl_omniglot_rln(input_rln)
    rln = tf.keras.Model(inputs=input_rln, outputs=h)
    y = mrcl_omniglot_tln(input_tln, output=classes)
    tln = tf.keras.Model(inputs=input_tln, outputs=y)
    return rln, tln


def get_data_by_classes(background_data, evaluation_data):
    background_data = np.array(sorted(list(tfds.as_numpy(background_data)), key=itemgetter('label')))
    evaluation_data = np.array(sorted(list(tfds.as_numpy(evaluation_data)), key=itemgetter('label')))
    background_training_data = []
    evaluation_training_data = []
    evaluation_test_data = []

    evaluation_number_of_classes = int(evaluation_data.shape[0] / 20)
    background_number_of_classes = int(background_data.shape[0] / 20)
    for i in range(evaluation_number_of_classes):
        evaluation_training_data.append(evaluation_data[i * 20:i * 20 + 15])
        evaluation_test_data.append(evaluation_data[i * 20 + 15:(i + 1) * 20])

    for i in range(background_number_of_classes):
        background_training_data.append(background_data[i * 20:(i + 1) * 20])
    return background_training_data, evaluation_training_data, evaluation_test_data


def partition_into_disjoint(data):
    number_of_classes = len(data)
    s_learn = list(range(0, int(number_of_classes / 2)))
    s_remember = list(range(int(number_of_classes / 2), number_of_classes))
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


def pretrain_classification_mrcl(x_traj, y_traj, x_rand, y_rand, rln, tln, classification_parameters):
    # Random reinitialization of last layer
    w = tln.layers[-1].weights[0]
    new_w = tln.layers[-1].kernel_initializer(shape=w.shape)
    tln.layers[-1].weights[0].assign(new_w)

    # Clone tln to preserve initial weights
    tln_initial = tf.keras.models.clone_model(tln)

    x_meta = tf.concat([x_rand, x_traj], axis=0)
    y_meta = tf.concat([y_rand, y_traj], axis=0)

    for x, y in zip(x_traj, y_traj):
        inner_update(x, y, rln, tln, classification_parameters)

    with tf.GradientTape(persistent=True) as theta_tape:
        outer_loss = compute_loss(x_meta, y_meta, rln, tln)

    tln_gradients = theta_tape.gradient(outer_loss, tln.trainable_variables)
    rln_gradients = theta_tape.gradient(outer_loss, rln.trainable_variables)
    del theta_tape

    classification_parameters["meta_optmizer"](
        learning_rate=classification_parameters["meta_learning_rate"]).apply_gradients(
        zip(tln_gradients + rln_gradients, tln_initial.trainable_variables + rln.trainable_variables))

    copy_parameters(tln_initial, tln)
    return outer_loss


@tf.function
def inner_update(x, y, rln, tln, classification_parameters):
    with tf.GradientTape(watch_accessed_variables=False) as Wj_Tape:
        Wj_Tape.watch(tln.trainable_variables)
        inner_loss = compute_loss(x, y, rln, tln, classification_parameters)
    gradients = Wj_Tape.gradient(inner_loss, tln.trainable_variables)
    for g, v in zip(gradients, tln.trainable_variables):
        v.assign_sub(classification_parameters["inner_learning_rate"] * g)


@tf.function
def compute_loss(x, y, rln, tln, classification_parameters):
    if x.shape.ndims == 3:
        output = tln(rln(tf.expand_dims(x, axis=0)))
    else:
        output = tln(rln(x))
    loss = classification_parameters["loss_function"](y, output)
    return loss
