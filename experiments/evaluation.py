import tensorflow as tf
import numpy as np


@tf.function
def compute_loss(x, y, loss_function, tln, rln):
    return loss_function(y, tln(rln(x)))


def train_and_evaluate(x_train, y_train, x_test, y_test, rln, tln, optimizer, loss_function, batch_size):
    results = {}
    for cls in range(len(x_train)):
        data = tf.data.Dataset.from_tensor_slices((x_train[cls], y_train[cls])).batch(batch_size)

        for x, y in data:
            with tf.GradientTape() as tape:
                loss = compute_loss(x, y, loss_function, tln, rln)
            gradient_tln = tape.gradient(loss, tln.trainable_variables)
            optimizer.apply_gradients(zip(gradient_tln, tln.trainable_variables))

        x_test_classes_seen = x_test[:cls + 1]
        y_test_classes_seen = y_test[:cls + 1]

        output_loss = 0
        i = 0
        for x, y in tf.data.Dataset.from_tensor_slices((x_test_classes_seen, y_test_classes_seen)):
            output_loss += compute_loss(x, y, loss_function, tln, rln)
            i += 1
        output_loss /= i

        results[cls] = {"loss": np.asscalar(np.array(output_loss)),
                        "number_of_classes_seen": cls + 1}

    return results

