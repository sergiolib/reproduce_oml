import tensorflow as tf

num_gb_to_use = 8
limit_gpu = False

if tf.test.is_gpu_available() and limit_gpu:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [
        tf.config.experimental.VirtualDeviceConfiguration(memory_limit=(1024 * num_gb_to_use))])
    print(f"Using GPU with {num_gb_to_use} GB memory")

classification_parameters = {
    "meta_learning_rate": 1e-4,
    "inner_learning_rate": 0.03,
    "loss_function": tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    "online_optimizer": tf.optimizers.SGD,
    "online_learning_rate": 0.001,
    "meta_optimizer": tf.optimizers.Adam
}

pretraining_parameters = {
    "loss_function": tf.losses.SparseCategoricalCrossentropy(from_logits=True)
}
