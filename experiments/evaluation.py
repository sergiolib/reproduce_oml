import tensorflow as tf
import numpy as np


@tf.function
def compute_loss(x, y, loss_function, rln, tln):
    return loss_function(y, tln(rln(x)))


def train_and_evaluate(x_train, y_train, x_test, y_test, rln, tln, optimizer, loss_function, batch_size):
    # Sample x_rand, y_rand from s_remember
    x_shape = x_train.shape
    x_train_f = tf.reshape(x_train, (x_shape[0] * x_shape[1], x_shape[2]))
    y_train_f = tf.reshape(y_train, (x_shape[0] * x_shape[1],))

    data = tf.data.Dataset.from_tensor_slices((x_train_f, y_train_f)).batch(batch_size)

    data_seen = 0

    results = {}

    for x, y in data:
        data_seen += 8
        with tf.GradientTape() as tape:
            loss = compute_loss(x, y, loss_function, rln, tln)
        gradient_tln = tape.gradient(loss, tln.trainable_variables)
        optimizer.apply_gradients(zip(gradient_tln, tln.trainable_variables))

        final_ind = data_in_test_seen(data_seen, x_train_f, x_test)
        x_test_seen_classes = x_test[:final_ind]
        y_test_seen_classes = y_test[:final_ind]

        results[data_seen] = {"loss": compute_loss(x_test_seen_classes, y_test_seen_classes, loss_function, rln, tln),
                              "tested_data_points": final_ind,
                              "data_points_trained_with": data_seen,
                              "number_of_classes_seen": len(np.unique(np.argmax(x_test_seen_classes[:, 1:], axis=1)))}

    return results


def data_in_test_seen(data_in_traj_seen, traj, rand):
    n_points_traj = len(traj)
    n_points_rand = len(rand)
    return int(n_points_rand / n_points_traj * (data_in_traj_seen + 1))
