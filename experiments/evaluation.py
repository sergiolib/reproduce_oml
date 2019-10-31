import tensorflow as tf


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
