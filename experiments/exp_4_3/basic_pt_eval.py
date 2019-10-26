import tensorflow as tf
import numpy as np


def basic_pt_eval(data, model, params, layers_to_freeze=1):
    batch_size = params["random_batch_size"]
    optimizer = params["optimizer"](learning_rate=params["learning_rate"])
    results = []

    # TODO: not sure if this works!!! this is probably deprecated in tf 2.0
    # create a var list of layers you want to train as trainable variables
    for layer_i in range(layers_to_freeze):
        model.get_layer("layer", layer_i).trainable = False

    for i in range(0, len(data["x"]), batch_size):
        batch = {a: data[a][i:i + batch_size] for a in data}

        with tf.GradientTape() as tape:
            outputs = model(batch["x"])
            loss = params["loss_metric"](outputs, batch["y"])

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        results += [np.array(outputs)]
    return np.concatenate(results)
