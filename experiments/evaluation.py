import tensorflow as tf
import numpy as np
from datasets.synth_datasets import gen_sine_data
from util.misc import factor_int


def compute_loss(x, y, loss_function, tln, rln):
    return loss_function(y, tln(rln(x)))


def train_and_evaluate(x_train, y_train, x_val, y_val, rln, tln, optimizer,
                       loss_function, batch_size, epochs=1):
    results_3a = {}
    results_3b = {}
    results_3a_tr = {}
    results_3b_tr = {}

    # For every class
    for cls in range(len(x_train)):
        for e in range(epochs):
            x_train_cls = x_train[cls]
            y_train_cls = y_train[cls]

            # get its data points
            data = tf.data.Dataset.from_tensor_slices(
                (x_train_cls, y_train_cls)).batch(batch_size)

            # Train with its data points using GD
            for x, y in data:
                with tf.GradientTape() as tape:
                    loss = compute_loss(x, y, loss_function, tln, rln)
                gradient_tln = tape.gradient(loss, tln.trainable_variables)
                optimizer.apply_gradients(zip(gradient_tln,
                                              tln.trainable_variables))

        # Calculate loss with seen data from training set
        x_train_classes_seen = tf.concat([i for i in x_train[:cls + 1]], 0)
        y_train_classes_seen = tf.concat([i for i in y_train[:cls + 1]], 0)

        training_loss = compute_loss(x_train_classes_seen,
                                     y_train_classes_seen,
                                     loss_function, tln, rln)
        results_3a_tr[cls + 1] = training_loss.numpy()

        # Calculate loss with unseen data from validation set
        x_val_classes_seen = tf.concat([i for i in x_val[:cls + 1]], 0)
        y_val_classes_seen = tf.concat([i for i in y_val[:cls + 1]], 0)

        validation_loss = compute_loss(x_val_classes_seen, y_val_classes_seen,
                                       loss_function, tln, rln)
        results_3a[cls + 1] = validation_loss.numpy()

    data_iter = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    for i, (x, y) in enumerate(data_iter):
        results_3b[i + 1] = compute_loss(x, y, loss_function,
                                         tln, rln).numpy()

    data_iter = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    for i, (x, y) in enumerate(data_iter):
        results_3b_tr[i + 1] = compute_loss(x, y, loss_function,
                                            tln, rln).numpy()

    validation_results = results_3a, results_3b
    training_results = results_3a_tr, results_3b_tr
    return training_results, validation_results


def evaluate_models_isw(x_train, y_train, x_val, y_val, tln, rln,
                        learning_rate, batch_size=8, epochs=1):
    loss_function = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

    results = train_and_evaluate(x_train=x_train, y_train=y_train,
                                 x_val=x_val, y_val=y_val, rln=rln,
                                 tln=tln, optimizer=optimizer,
                                 loss_function=loss_function,
                                 batch_size=batch_size, epochs=epochs)
    results_tr, results_val = results
    loss_per_class_during_training_val, interference_losses_val = results_val
    loss_per_class_during_training_tr, interference_losses_tr = results_tr
    mean_loss_all_val = sum(
        [i for i in interference_losses_val.values()]) / \
        len(interference_losses_val)
    mean_loss_all_tr = sum(
        [i for i in interference_losses_tr.values()]) / \
        len(interference_losses_tr)
    training_losses = (mean_loss_all_tr, loss_per_class_during_training_tr,
                       interference_losses_tr)
    validation_losses = (mean_loss_all_val, loss_per_class_during_training_val,
                         interference_losses_val)
    return training_losses, validation_losses


def prepare_data_evaluation(tasks, n_functions, sample_length, repetitions,
                            seed=None):

    data = gen_sine_data(tasks, n_functions, sample_length, repetitions,
                         seed=seed)

    x_train_f_r_s_x, y_train_f_r_s, x_val_f_s_x, y_val_f_s = data

    # Reshape for inputting to training method
    x_train_r_s_f_x = np.transpose(x_train_f_r_s_x, (1, 2, 0, 3))
    y_train_r_s_f = np.transpose(y_train_f_r_s, (1, 2, 0))
    x_train_rs_f_x = np.reshape(x_train_r_s_f_x, (repetitions * sample_length,
                                                  n_functions, -1))
    y_train_rs_f = np.reshape(
        y_train_r_s_f, (repetitions * sample_length, n_functions))
    x_train_f_rs_x = np.transpose(x_train_rs_f_x, (1, 0, 2))
    y_train_f_rs = np.transpose(y_train_rs_f, (1, 0))

    # Numpy -> Tensorflow
    x_train_f_rs_x = tf.convert_to_tensor(x_train_f_rs_x, dtype=tf.float32)
    y_train_f_rs = tf.convert_to_tensor(y_train_f_rs, dtype=tf.float32)
    x_val_f_s_x = tf.convert_to_tensor(x_val_f_s_x, dtype=tf.float32)
    y_val_f_s = tf.convert_to_tensor(y_val_f_s, dtype=tf.float32)

    return x_train_f_rs_x, y_train_f_rs, x_val_f_s_x, y_val_f_s


def compute_sparsity(x, rln, tln):
    rep = rln(x)
    rep = np.array(rep)
    counts = np.isclose(rep, 0).sum(axis=1) / rep.shape[1]
    sparsity = np.mean(counts)
    return sparsity


def get_representations_graphics(x, rln):
    x = tf.reshape(x, [-1, 11])
    rep = rln(x)
    rep_len = rep.shape[-1]
    rep_f1, rep_f2 = factor_int(rep_len)
    rep = [tf.reshape(r, (rep_f1, rep_f2, 1)) for r in rep]
    rep = [r / tf.reduce_max(r) for r in rep]
    rep = tf.random.shuffle(tf.stack(rep))
    return rep
