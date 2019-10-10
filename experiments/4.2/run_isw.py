regression_parameters = {
    "meta_learning_rate": 1e-4,  # alpha
    "inner_learning_rate": 3e-3,  # beta
    "loss_metric": tf.losses.MSE,
    "total_gradient_updates": 40,  # n # TODO: check this as it is not explicit in the paper
    "inner_steps": 400, # k
    "inner_optimizer": tf.optimizers.SGD,
    "meta_optimizer": tf.optimizers.Adam,
    "random_batch_size": 8  # len(X_rand)
}
partition = synth_datasets.partition_sine_data(synth_datasets.gen_sine_data(n_id=900))
pretraining = partition["pretraining"]
evaluation = partition["evaluation"]

s_learn, s_remember = split_data_in_2(pretraining, 0.5)

# Insert data in Tensoflow
s_learn["z"] = tf.convert_to_tensor(s_learn["z"], dtype=tf.float32)
s_learn["k"] = tf.convert_to_tensor(s_learn["k"], dtype=tf.int32)
s_learn["y"] = tf.convert_to_tensor(s_learn["y"], dtype=tf.float32)

s_remember["z"] = tf.convert_to_tensor(s_remember["z"], dtype=tf.float32)
s_remember["k"] = tf.convert_to_tensor(s_remember["k"], dtype=tf.int32)
s_remember["y"] = tf.convert_to_tensor(s_remember["y"], dtype=tf.float32)

# Make k be in one hot representation
s_learn["k"] = tf.one_hot(s_learn["k"], depth=900)
s_remember["k"] = tf.one_hot(s_remember["k"], depth=900)

# Join X values
learn_z = tf.reshape(s_learn["z"], (-1, 1))
learn_x = tf.concat([learn_z, s_learn["k"]], axis=1)
learn_y = s_learn["y"]

remember_z = tf.reshape(s_remember["z"], (-1, 1))
remember_x = tf.concat([remember_z, s_remember["k"]], axis=1)
remember_y = s_remember["y"]

rln, tln = mrcl_isw(one_hot_depth=900)  # Actual model

mrcl_pretrain((learn_x, learn_y), (remember_x, remember_y), rln, tln, regression_parameters)
