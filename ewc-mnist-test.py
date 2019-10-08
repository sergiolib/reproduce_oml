import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Dense, Flatten, BatchNormalization
from tensorflow.keras import Model
import sys
from ewc import ElasticWeightConsolidation

tf.keras.backend.set_floatx('float64')

model = ElasticWeightConsolidation(28 * 28, 100, 10)

batch_size = 100
loss_object = model.loss
optimizer = tf.keras.optimizers.Adam(lr=1e-4)
EPOCHS = 5

# Prepare mnist dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train = np.array(y_train, dtype=np.int32)
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size).shuffle(100000)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

# Prepare fashion mnist dataset
f_mnist = tf.keras.datasets.fashion_mnist
(f_x_train, f_y_train), (f_x_test, f_y_test) = f_mnist.load_data()
f_x_train, f_x_test = f_x_train / 255.0, f_x_test / 255.0
f_train_ds = tf.data.Dataset.from_tensor_slices((f_x_train, f_y_train)).batch(batch_size).shuffle(100000, reshuffle_each_iteration=True)
f_test_ds = tf.data.Dataset.from_tensor_slices((f_x_test, f_y_test)).batch(batch_size)

# Define loss/accuracy metrics
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):
        predictions = model(images, training=False)
        t_loss = loss_object(labels, predictions)

        test_loss(t_loss)
        test_accuracy(labels, predictions)

for epoch in range(EPOCHS):
    for images, labels in train_ds:
        train_step(images, labels)

    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch+1,
                        train_loss.result(),
                        train_accuracy.result()*100,
                        test_loss.result(),
                        test_accuracy.result()*100))

    # Reset the metrics for the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

model.register_ewc_params(tf.data.Dataset.from_tensor_slices((x_train, y_train)), 300)

for epoch in range(EPOCHS):
    for images, labels in f_train_ds:
        train_step(images, labels)

    for test_images, test_labels in f_test_ds:
        test_step(test_images, test_labels)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch+1,
                        train_loss.result(),
                        train_accuracy.result()*100,
                        test_loss.result(),
                        test_accuracy.result()*100))

    # Reset the metrics for the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

model.register_ewc_params(tf.data.Dataset.from_tensor_slices((f_x_train, np.array(f_y_train, dtype=np.int32))), 300)

test_accuracy.reset_states()
for test_images, test_labels in test_ds:
    test_step(test_images, test_labels)
print('Test accuracy on mnist', test_accuracy.result().numpy())
test_accuracy.reset_states()
for test_images, test_labels in f_test_ds:
    test_step(test_images, test_labels)
print('Test accuracy on fashion mnist', test_accuracy.result().numpy())