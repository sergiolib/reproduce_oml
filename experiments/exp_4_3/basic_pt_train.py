import numpy as np
import tensorflow as tf
import tqdm
import os

from datasets.synth_datasets import gen_sine_data


def basic_pt_train(params, data, model):
    """
    Train a simple NN
    """
    # Main loop
    trange = tqdm.tqdm(range(params["total_gradient_updates"]))
    # Initialize optimizer
    optimizer = params["optimizer"](learning_rate=params["learning_rate"])

    # k is the amount of samples of each class
    k = int(len(data['x'][0]) / 10)

    for i in trange:  # each of the 40 optimizations
        for n in range(10):  # each of the 10 classes
            x_traj, y_traj, x_rand, y_rand = gen_sine_data(tasks=tasks, n_functions=params["n_functions"],
                                                           sample_length=params["sample_length"],
                                                           repetitions=params["repetitions"])

            # Reshape for inputting to training method
            x_traj = np.vstack(x_traj)
            y_traj = np.vstack(y_traj)
            x_rand = np.vstack(x_rand)
            y_rand = np.hstack(y_rand)

            # Numpy -> Tensorflow
            x_rand = tf.convert_to_tensor(x_rand, dtype=tf.float32)
            y_rand = tf.convert_to_tensor(y_rand, dtype=tf.float32)
            x_traj = tf.convert_to_tensor(x_traj, dtype=tf.float32)
            y_traj = tf.convert_to_tensor(y_traj, dtype=tf.float32)

            # Random reinitialization TODO: we dont need this here right?
            # w = model.layers[-1].weights[0]
            # new_w = model.layers[-1].kernel_initializer(shape=w.shape)
            # model.layers[-1].weights[0].assign(new_w)

            # Sample x_rand, y_rand from the dataset
            # TODO: fix sampling from dictionary
            x_rand, y_rand = random_sample(data, params["random_batch_size"])

            with tf.GradientTape() as tape:
                outputs = model(x_rand)
                loss = params["loss_metric"](outputs, y_rand)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # loss = tf.reduce_mean(params["loss_metric"](model(data[0]), data[1]))
        # trange.set_description(f"{loss}")

        if i > 0 and i % 10 == 0:
            try:
                os.path.isdir("saved_models/")
            except NotADirectoryError:
                os.makedirs("save_models")
            model.save(f"saved_models/basic_pt_{i}.tf", save_format="tf")

    # Save final
    try:
        os.path.isdir("saved_models/")
    except NotADirectoryError:
        os.makedirs("save_models")
    model.save(f"saved_models/basic_pt_final.tf", save_format="tf")




run_basic_pt()
