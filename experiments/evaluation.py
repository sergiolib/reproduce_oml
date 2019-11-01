import tensorflow as tf
import numpy as np
from datasets.synth_datasets import gen_sine_data
from util.misc import factor_int


@tf.function
def compute_loss(x, y, loss_function, tln, rln):
    return loss_function(y, tln(rln(x)))


def train_and_evaluate(x_train, y_train, x_val, y_val, rln, tln, optimizer,
                       loss_function, batch_size, epochs=1):
    # For every class
    results_3a = {}
    results_3b = {}
    for cls in range(len(x_train)):
        prev_loss = float("inf")
        for e in range(epochs):
            # get its data points
            data = tf.data.Dataset.from_tensor_slices((x_train[cls], y_train[cls])).batch(batch_size)

            # Train with its data points using GD
            for x, y in data:
                with tf.GradientTape() as tape:
                    loss = compute_loss(x, y, loss_function, tln, rln)
                gradient_tln = tape.gradient(loss, tln.trainable_variables)
                optimizer.apply_gradients(zip(gradient_tln,
                                              tln.trainable_variables))

            # Validate with unseen data
            x_val_classes_seen = tf.expand_dims(x_val[cls], 0)
            y_val_classes_seen = tf.expand_dims(y_val[cls], 0)

            output_loss = 0
            i = 0

            for x, y in tf.data.Dataset.from_tensor_slices((x_val_classes_seen, y_val_classes_seen)):
                output_loss += compute_loss(x, y, loss_function, tln, rln)
                i += 1
            output_loss /= i

            # Early stopping
            if prev_loss < output_loss:
                break
            else:
                # t.set_description(f"Class: {cls}\tClass loss: {output_loss:.4}")
                prev_loss = output_loss

        # Class trained: evaluate all seen data so far
        x_val_classes_seen = x_val[:cls + 1]
        y_val_classes_seen = y_val[:cls + 1]

        output_loss = 0
        data_iter = tf.data.Dataset.from_tensor_slices((x_val_classes_seen,
                                                        y_val_classes_seen))
        i = 0
        for x, y in data_iter:
            output_loss += float(compute_loss(x, y, loss_function, tln, rln))
            i += 1
        output_loss /= i
        results_3a[cls + 1] = output_loss

    x_val_classes_seen = x_val
    y_val_classes_seen = y_val

    data_iter = tf.data.Dataset.from_tensor_slices((x_val_classes_seen,
                                                    y_val_classes_seen))
    for i, (x, y) in enumerate(data_iter):
        results_3b[i + 1] = float(compute_loss(x, y, loss_function, tln, rln))

    # Results_3a: Mean Squared Error of classes seen so far
    # Results_3b: Mean Squared Error of each class in the end
    return results_3a, results_3b


def evaluate_models_isw(x_train, y_train, x_val, y_val, tln, rln,
                        reset_last_layer=True, batch_size=8, epochs=1):
    loss_function = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.SGD()

    if reset_last_layer:
        # Random reinitialization of last layer
        w = tln.layers[-1].weights[0]
        b = tln.layers[-1].weights[1]
        new_w = tf.keras.initializers.he_normal()(shape=w.shape)
        w.assign(new_w)
        new_b = tf.keras.initializers.zeros()(shape=b.shape)
        b.assign(new_b)
    results = train_and_evaluate(x_train=x_train, y_train=y_train,
                                 x_val=x_val, y_val=y_val, rln=rln,
                                 tln=tln, optimizer=optimizer,
                                 loss_function=loss_function,
                                 batch_size=batch_size, epochs=epochs)
    loss_per_class_during_training, interference_losses = results
    mean_loss_all_val = sum([i for i in interference_losses.values()]) / len(interference_losses)
    return mean_loss_all_val, loss_per_class_during_training, interference_losses


def prepare_data_evaluation(tasks, n_functions, sample_length, repetitions):
    x_train, y_train, x_val, y_val = gen_sine_data(tasks, n_functions,
                                                   sample_length,
                                                   repetitions)

    # Reshape for inputting to training method
    x_train = np.transpose(x_train, (1, 2, 0, 3))
    y_train = np.transpose(y_train, (1, 2, 0))
    x_train = np.reshape(x_train, (repetitions * sample_length, n_functions, -1))
    y_train = np.reshape(y_train, (repetitions * sample_length, n_functions))
    x_train = np.transpose(x_train, (1, 0, 2))
    y_train = np.transpose(y_train, (1, 0))

    # Numpy -> Tensorflow
    x_val = tf.convert_to_tensor(x_val, dtype=tf.float32)
    y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)
    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)

    return x_train, y_train, x_val, y_val

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
