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
            split[k] = s[ordered_inds[i]]
        res.append(split)
    return res

def copy_parameters(source, dest):
    """Copies parameters of source model into the destination one"""
    for matrix_source, matrix_dest in zip(source.weights, dest.weights):
        matrix_dest.assign(matrix_source)

def sample_trajectory(data_list, start, size):
    ret = []
    for d in data_list:
        ret.append(tf.reshape(d[start:start+size], (size, -1)))
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

def mrcl_pretrain(s_learn, s_remember, rln, tln, params):
    # Create auxiliary model
    _, tln_inner = mrcl_isw(one_hot_depth=900)  # For inner calculations
    
    # Main loop
    trange = tqdm.tqdm(range(params["total_gradient_updates"]))
    for i in trange:
        # Copy parameters of tln to inner tln
        copy_parameters(tln, tln_inner)

        # Random reinitialization
        w = tln.layers[-1].weights[0]
        new_w = tln.layers[-1].kernel_initializer(shape=w.shape)
        tln.layers[-1].weights[0].assign(new_w)
        
        # Sample x_traj, y_traj from S_learn
        x_traj, y_traj = sample_trajectory(s_learn, i * params["inner_steps"], 
                                   params["inner_steps"])
        for j in range(params["inner_steps"]):
            with tf.GradientTape() as tape:
                representations = rln(x_traj)
                outputs = tln_inner(representations)
                loss = params["loss_metric"](outputs, y_traj)
                gradients = tape.gradient(loss, tln_inner.trainable_variables)
                params["inner_optimizer"]().apply_gradients(zip(gradients, tln_inner.trainable_variables))

        # Sample x_rand, y_rand from s_remember
        x_rand, y_rand = random_sample(s_remember, params["random_batch_size"])
        x_meta = tf.concat([x_rand, x_traj], axis=0)
        y_meta = tf.concat([y_rand, y_traj], axis=0)

        with tf.GradientTape() as tape:
            representations = rln(x_traj)
            outputs = tln_inner(representations)
            loss = params["loss_metric"](outputs, y_traj)
            gradients = tape.gradient(loss, tln_inner.trainable_variables)
            params["meta_optimizer"]().apply_gradients(zip(gradients, tln.trainable_variables))
