import os
import sys
sys.path.append("")
import tqdm
import tensorflow as tf
import numpy as np


def split_data_in_2(data_dict, proportion=0.9):
    """
    Split the data dictionary in 2 disjoint sets, in which the first one
    has the given proportion of the data while the other 1 - the given value.
    """
    res = []
    number_of_samples = len(data_dict["y"])
    random_order_inds = np.random.permutation(number_of_samples)
    split_point = int(proportion * number_of_samples)
    ordered_inds = np.sort(random_order_inds[:split_point]), np.sort(random_order_inds[split_point:])
    for i in range(2):
        split = {}
        for k, s in data_dict.items():
            dtype = s.dtype
            s = np.array(s)
            split[k] = tf.convert_to_tensor(s[ordered_inds[i]], dtype=dtype)
        res.append(split)
    return res


def copy_parameters(source, dest):
    """Copies parameters of source model into the destination one"""
    for matrix_source, matrix_dest in zip(source.weights, dest.weights):
        matrix_dest.assign(matrix_source)


def sample_trajectory(data_list, start, size):
    ret = []
    inds = np.random.permutation(size)
    for d in data_list:
        assert start < len(d), "Start value is incorrect for the inputs given"
        assert len(d) >= start + size, "Provided size is incompatible with the requested size"
        v = tf.reshape(d[start:start+size], (size, -1))
        v = tf.convert_to_tensor(np.array(v)[inds])
        ret.append(v)
    return tuple(ret)


def random_sample(data_list, batch_size):
    results = []
    for d in data_list:
        l = []
        for i in np.random.permutation(len(d))[:batch_size]:
            l.append(tf.reshape(d[i], (1, -1)))
        l = tf.concat(l, axis=0)
        results.append(l)
    return tuple(results)


def sample_trajectory_wrap(data, start, size):
    x_len = len(data[0])
    space_left_forward = x_len - start

    x_traj, y_traj = sample_trajectory(data, start,
                                       min(space_left_forward, size))
    if space_left_forward < size:
        x_traj_2, y_traj_2 = sample_trajectory(data, 0, size - space_left_forward)
        x_traj = tf.concat([x_traj, x_traj_2], axis=0)
        y_traj = tf.concat([y_traj, y_traj_2], axis=0)
    return x_traj, y_traj


def mrcl_pretrain(s_learn, s_remember, rln, tln, params):
    # Main loop
    trange = tqdm.tqdm(range(params["total_gradient_updates"]))
    # Initialize optimizer
    meta_optimizer_inner = params["meta_optimizer"](learning_rate=params["inner_learning_rate"])
    meta_optimizer_outer = params["meta_optimizer"](learning_rate=params["meta_learning_rate"])
    loss_fun = params["loss_metric"]

    for i in trange:  # each of the 40 optimizations
        # Random reinitialization of last layer
        # w = tln.layers[-1].weights[0]
        # new_w = tln.layers[-1].kernel_initializer(shape=w.shape)
        # tln.layers[-1].weights[0].assign(new_w)

        # Clone tln to preserve initial weights
        tln_initial = tf.keras.models.clone_model(tln)

        # Sample x_traj, y_traj from S_learn
        x_traj, y_traj = sample_trajectory_wrap(s_learn, (i * params["inner_steps"]) % len(s_learn[0]), params["inner_steps"])

        with tf.GradientTape(watch_accessed_variables=False) as W_Tape:
            W_Tape.watch(tln.trainable_variables)
            for j in range(params["inner_steps"]):
                with tf.GradientTape(watch_accessed_variables=False) as Wj_Tape:
                    Wj_Tape.watch(tln.trainable_variables)
                    output = tln(rln(tf.reshape(x_traj[j], (1, -1))))
                    inner_loss = loss_fun(output, y_traj[j])
                gradients = Wj_Tape.gradient(inner_loss, tln.trainable_variables)
                meta_optimizer_inner.apply_gradients(zip(gradients, tln.trainable_variables))

            # Sample x_rand, y_rand from s_remember
            x_rand, y_rand = random_sample(s_remember, params["random_batch_size"])
            x_meta = tf.concat([x_rand, x_traj], axis=0)
            y_meta = tf.concat([y_rand, y_traj], axis=0)
            inds = np.random.permutation(len(x_meta))
            x_meta = tf.gather(x_meta, inds)
            y_meta = tf.gather(y_meta, inds)

            with tf.GradientTape(watch_accessed_variables=False) as theta_Tape:
                theta_Tape.watch(rln.trainable_variables)
                outputs = tln(rln(x_meta))
                outer_loss = loss_fun(outputs, y_meta)

        tln_gradients = W_Tape.gradient(outer_loss, tln.trainable_variables)
        # prv = float(tln_initial.trainable_variables[0][0, 0])
        meta_optimizer_outer.apply_gradients(zip(tln_gradients, tln_initial.trainable_variables))
        # aft = float(tln_initial.trainable_variables[0][0, 0])
        # grd = float(tln_gradients[0][0, 0])
        # if grd > 0:
        #     assert aft != prv
        rln_gradients = theta_Tape.gradient(outer_loss, rln.trainable_variables)
        meta_optimizer_outer.apply_gradients(zip(rln_gradients, rln.trainable_variables))

        copy_parameters(tln_initial, tln)

        loss = tf.reduce_mean(params["loss_metric"](tln(rln(s_remember[0])), s_remember[1]))
        trange.set_description(f"{loss}")

        if i > 0 and i % 10 == 0:
            try:
                os.path.isdir("saved_models/")
            except NotADirectoryError:
                os.makedirs("save_models")
            rln.save(f"saved_models/rln_{i}.tf", save_format="tf")
            tln.save(f"saved_models/tln_{i}.tf", save_format="tf")

    # Save final
    try:
        os.path.isdir("saved_models/")
    except NotADirectoryError:
        os.makedirs("save_models")
    rln.save(f"saved_models/rln_final.tf", save_format="tf")
    tln.save(f"saved_models/tln_final.tf", save_format="tf")


def mrcl_evaluate(data, rln, tln, params):
    batch_size = params["random_batch_size"]
    optimizer = params["online_optimizer"](learning_rate=params["inner_learning_rate"])
    loss_fun = params["loss_metric"]
    results = []
    for i in range(0, len(data["x"]), batch_size):
        batch = {a: data[a][i:i + batch_size] for a in data}
        with tf.GradientTape() as tape:
            outputs = tln(rln(batch["x"]))
            loss = loss_fun(outputs, batch["y"])
        tln_gradients = tape.gradient(loss, tln.trainable_variables)
        optimizer.apply_gradients(zip(tln_gradients, tln.trainable_variables))
        results += [np.array(outputs)]
    return np.concatenate(results)
