import tensorflow as tf
import datetime
from experiments.exp4_2.omniglot_model import mrcl_omniglot, get_data_by_classes, evaluate_classification_mrcl
from datasets.tf_datasets import load_omniglot
import os
import json

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7168)])
print(f"GPU is available: {tf.test.is_gpu_available()}")

dataset = "omniglot"
background_data, evaluation_data = load_omniglot(dataset, verbose=1)
background_training_data, evaluation_training_data, evaluation_test_data, background_training_data_15, background_training_data_5 = get_data_by_classes(background_data,
                                                                                               evaluation_data)
assert len(evaluation_training_data) == 659
assert len(evaluation_test_data) == 659
assert len(background_training_data) == 964

assert evaluation_training_data[0].shape[0] == 15
assert evaluation_test_data[0].shape[0] == 5
assert background_training_data[0].shape[0] == 20

classification_parameters = {
    "meta_learning_rate": 1e-4,
    "inner_learning_rate": 0.03,
    "loss_function": tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    "online_optimizer": tf.optimizers.SGD,
    "online_learning_rate": 0.001,
    "meta_optimizer": tf.optimizers.Adam
}


current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/classification/gradient_tape/' + current_time + '/train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

rln, tln = mrcl_omniglot(200)

try:
    os.stat("evaluation_results_scratch_omniglot")
except:
    os.mkdir("evaluation_results_scratch_omniglot")

original_rln, original_tln = mrcl_omniglot(10)
points = [10, 50, 75, 100, 150, 200]
for point in points:
    lrs = [0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001]
    all_results = {}
    for lr in lrs:
        classification_parameters["online_learning_rate"] = lr
        rln, tln = mrcl_omniglot(200)
        tln.set_weights(original_tln.get_weights())
        rln.set_weights(original_rln.get_weights())
        all_results[f"{lr}"] = evaluate_classification_mrcl(evaluation_training_data, evaluation_test_data, rln, tln, point, classification_parameters)
        lr_str = f"{lr}".replace(".", "_")
        with open(f"evaluation_results/mrcl_omniglot_{lr_str}_{point}.json", 'w') as f:  # writing JSON object
            json.dump(all_results[f"{lr}"], f)
