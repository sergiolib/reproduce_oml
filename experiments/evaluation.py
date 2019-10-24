import tensorflow as tf
import numpy as np


@tf.function
def compute_loss(x, y, loss_function, rln, tln):
    return loss_function(y, tln(rln(x)))


def train_and_evaluate(x_traj, y_traj, x_rand, y_rand, rln, tln, optimizer, loss_function, batch_size):
    # Sample x_rand, y_rand from s_remember
    x_shape = x_traj.shape
    x_traj_f = tf.reshape(x_traj, (x_shape[0] * x_shape[1], x_shape[2]))
    y_traj_f = tf.reshape(y_traj, (x_shape[0] * x_shape[1],))

    data = tf.data.Dataset.from_tensor_slices((x_traj_f, y_traj_f)).shuffle(13000).batch(batch_size)

    data_seen = 0

    results = {}

    for x, y in data:
        data_seen += 8
        with tf.GradientTape() as tape:
            loss = compute_loss(x, y, loss_function, rln, tln)
        gradient_tln = tape.gradient(loss, tln.trainable_variables)
        optimizer.apply_gradients(zip(gradient_tln, tln.trainable_variables))

        final_ind = data_in_meta_seen(data_seen, x_traj_f, x_rand)
        x_rand_seen_classes = x_rand[:final_ind]
        y_rand_seen_classes = y_rand[:final_ind]

        results[data_seen] = {"loss": compute_loss(x_rand_seen_classes, y_rand_seen_classes, loss_function, rln, tln),
                              "tested_data_points": final_ind,
                              "data_points_trained_with": data_seen,
                              "number_of_classes_seen": len(np.unique(np.argmax(x_rand_seen_classes[:, 1:], axis=1)))}

    return results


def data_in_meta_seen(data_in_traj_seen, traj, rand):
    n_points_traj = len(traj)
    n_points_rand = len(rand)
    return int(n_points_rand / n_points_traj * (data_in_traj_seen + 1))