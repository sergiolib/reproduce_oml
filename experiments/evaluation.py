import tensorflow as tf
import numpy as np
import tqdm


@tf.function
def compute_loss(x, y, loss_function, tln, rln):
    return loss_function(y, tln(rln(x)))


def train_and_evaluate(x_train, y_train, x_val, y_val, rln, tln, optimizer,
                       loss_function, batch_size, epochs=1000):
    results = {}

    # For every class
    for cls in range(len(x_train)):
        prev_loss = float("inf")
        t = tqdm.trange(epochs)
        for e in t:
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
            x_val_classes_seen = x_val[:cls + 1]
            y_val_classes_seen = y_val[:cls + 1]

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
                t.set_description(f"Class: {cls}\tLoss: {output_loss:.4}")
                prev_loss = output_loss

        results[cls + 1] = np.asscalar(np.array(prev_loss))

    return results
