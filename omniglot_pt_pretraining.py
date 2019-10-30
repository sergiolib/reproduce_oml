import tensorflow as tf
import datetime
import numpy as np

from experiments.exp4_2.omniglot_model import mrcl_omniglot, get_background_data_by_classes, get_eval_data_by_classes, pre_train
from datasets.tf_datasets import load_omniglot
from experiments.training import save_models
from parameters import pretraining_parameters

print(f"GPU is available: {tf.test.is_gpu_available()}")

background_data, _ = load_omniglot(verbose=1)
_, background_training_data_15, background_training_data_5 = get_background_data_by_classes(background_data)
x_training = []
y_training = []
for class_id in range(len(background_training_data_15)):
    for training_item in background_training_data_15[class_id]:
        x_training.append(training_item['image'])
        y_training.append(training_item['label'])
x_training = tf.convert_to_tensor(x_training)
y_training = tf.convert_to_tensor(y_training)

x_testing = []
y_testing = []
for class_id in range(len(background_training_data_5)):
    for training_item in background_training_data_5[class_id]:
        x_testing.append(training_item['image'])
        y_testing.append(training_item['label'])
x_testing = tf.convert_to_tensor(x_testing)
y_testing = tf.convert_to_tensor(y_testing)

t = range(100)
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

learning_rates = [0.01, 0.001, 0.0001, 0.00001]
for lr in learning_rates:
    rln, tln = mrcl_omniglot()
    train_log_dir = f'logs/omniglot_{lr}/' + current_time + '/pre_train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    for epoch, v in enumerate(t):
        for x, y in tf.data.Dataset.from_tensor_slices((x_training, y_training)).shuffle(True).batch(32):
            loss, _ = pre_train(x, y, rln, tln, lr, pretraining_parameters)
        print("learning rate:", lr, "Epoch:", epoch, "Training loss:", loss.numpy())
        with train_summary_writer.as_default():
            tf.summary.scalar('Training loss', loss, step=epoch)
    
        if epoch % 10 == 0 and epoch != 0:
            total_correct = 0
            for x, y in tf.data.Dataset.from_tensor_slices((x_training, y_training)).shuffle(True).batch(32):
                loss, output = pre_train(x, y, rln, tln, lr, pretraining_parameters)
                after_softmax = tf.nn.softmax(output, axis=1)
                correct_prediction = tf.equal(tf.cast(tf.argmax(after_softmax, axis=1), tf.int32), y)
                total_correct = total_correct + tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
            train_accuracy = total_correct / x_training.shape[0]

            total_correct = 0
            for x, y in tf.data.Dataset.from_tensor_slices((x_testing, y_testing)).shuffle(True).batch(32):
                loss, output = pre_train(x, y, rln, tln, lr, pretraining_parameters)
                after_softmax = tf.nn.softmax(output, axis=1)
                correct_prediction = tf.equal(tf.cast(tf.argmax(after_softmax, axis=1), tf.int32), y)
                total_correct = total_correct + tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
            test_accuracy = total_correct / x_testing.shape[0]

            with train_summary_writer.as_default():
                tf.summary.scalar('Training accuracy', train_accuracy, step=epoch)
                tf.summary.scalar('Testing accuracy', test_accuracy, step=epoch)

            print("Epoch:", epoch, "Training loss:", loss.numpy(), "Training accuracy:", train_accuracy.numpy(), "Testing accuracy:", test_accuracy.numpy())
        if (epoch+1) % 1000 == 0:
            save_models(tln, f"tln_basic_pretraining_{epoch}_{lr}_omniglot")
            save_models(rln, f"rln_basic_pretraining_{epoch}_{lr}_omniglot")
