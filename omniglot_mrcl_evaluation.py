import tensorflow as tf
import datetime
import numpy as np
from experiments.exp4_2.omniglot_model import mrcl_omniglot, get_data_by_classes, \
    evaluate_classification_mrcl
from datasets.tf_datasets import load_omniglot
import os
import json


def evaluate(sort_samples=True, model_name="rln_pretraining_mrcl_1900_omniglot.tf"):

    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # tf.config.experimental.set_virtual_device_configuration(gpus[0], [
    #     tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7168)])
    # print(f"GPU is available: {tf.test.is_gpu_available()}")

    dataset = "omniglot"
    background_data, evaluation_data = load_omniglot(dataset, verbose=1)
    background_training_data, evaluation_training_data, evaluation_test_data, \
    background_training_data_15, background_training_data_5 = get_data_by_classes(background_data, evaluation_data, sort=sort_samples)

    classification_parameters = {
        "meta_learning_rate": 1e-4,
        "inner_learning_rate": 0.03,
        "loss_function": tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        "online_optimizer": tf.optimizers.SGD,
        "online_learning_rate": 0.001,
        "meta_optimizer": tf.optimizers.Adam
    }

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    rln_saved = tf.keras.models.load_model("saved_models_300_nodes/" + model_name)
    try:
        os.stat("evaluation_results")
    except IOError:
        os.mkdir("evaluation_results")

    points = [10, 50, 75, 100, 150, 200]
    for point in points:
        _, original_tln = mrcl_omniglot(point)
        lrs = [0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001]
        test_accuracy_results = []
        train_accuracy_results = []
        for lr in lrs:
            classification_parameters["online_learning_rate"] = lr
            rln, tln = mrcl_omniglot(point)
            tln.set_weights(original_tln.get_weights())
            rln.set_weights(rln_saved.get_weights())
            test_accuracy, train_accuracy = evaluate_classification_mrcl(evaluation_training_data, evaluation_test_data,
                                                                         rln, tln, point, classification_parameters)
            test_accuracy_results.append(test_accuracy)
            train_accuracy_results.append(train_accuracy)
            print(f"Learning rate {lr}, test accuracy {test_accuracy}, train accuracy {train_accuracy}")

        test_lr = lrs[np.argmax(np.array(test_accuracy_results))]
        train_lr = lrs[np.argmax(np.array(train_accuracy_results))]
        print(
            f"Number of classes {point}. Best testing learning rate is {test_lr} and best training learning rate is {train_lr}.")
        test_accuracy_results = []
        train_accuracy_results = []

        print(f"Starting 50 iterations of evaluation testing with learning rate {test_lr}.")
        for _ in range(50):
            classification_parameters["online_learning_rate"] = test_lr
            rln, tln = mrcl_omniglot(point)
            tln.set_weights(original_tln.get_weights())
            rln.set_weights(rln_saved.get_weights())
            test_accuracy, _ = evaluate_classification_mrcl(evaluation_training_data, evaluation_test_data, rln, tln,
                                                            point, classification_parameters)
            test_accuracy_results.append(str(test_accuracy))
        with open(f"evaluation_results/mrcl_omniglot_testing_{point}.json",
                  'w') as f:  # writing JSON object
            json.dump(test_accuracy_results, f)

        print(f"Starting 50 iterations of evaluation training with learning rate {train_lr}.")
        for _ in range(50):
            classification_parameters["online_learning_rate"] = train_lr
            rln, tln = mrcl_omniglot(point)
            tln.set_weights(original_tln.get_weights())
            rln.set_weights(rln_saved.get_weights())
            _, train_accuracy = evaluate_classification_mrcl(evaluation_training_data, evaluation_test_data, rln,
                                                             tln, point, classification_parameters)
            train_accuracy_results.append(str(train_accuracy))
        with open(f"evaluation_results/mrcl_omniglot_training_{point}.json",
                  'w') as f:  # writing JSON object
            json.dump(train_accuracy_results, f)

evaluate(sort_samples=True, model_name="rln_pretraining_mrcl_1900_omniglot.tf")
