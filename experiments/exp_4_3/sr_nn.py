import tensorflow as tf
import datetime
import numpy as np
from experiments.exp4_2.omniglot_model import mrcl_omniglot, get_data_by_classes, \
    partition_into_disjoint, pretrain_classification_mrcl, sample_trajectory, sample_random, sample_random_10_classes
from datasets.tf_datasets import load_omniglot
from experiments.training import save_models

# Their parameter settings
# argparser.add_argument('--epoch', type=int, help='epoch number', default=30)
# argparser.add_argument('--beta', type=float, help='epoch number', default=0.1)
# argparser.add_argument('--seed', type=int, help='epoch number', default=222)
# argparser.add_argument('--dataset', help='Name of experiment', default="omniglot")
# argparser.add_argument("--l1", action="store_true")
# argparser.add_argument('--lr', type=float, help='task-level inner update learning rate', default=0.0001)
# argparser.add_argument('--classes', type=int, nargs='+', help='Total classes to use in training',
#                        default=[0, 1, 2, 3, 4])
# argparser.add_argument('--name', help='Name of experiment', default="baseline")


params = {
    "lr": 0.0001,
    "loss_function": tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    "optimizer": tf.optimizers.Adam
}


def sr_nn_model(n_conv_layers=6, n_fc_layers=2, filters=256, h_units=300, output=964, strides=[2, 1, 2, 1, 2, 2]):
    input_layer = tf.keras.Input(shape=(84, 84, 1))

    h = input_layer
    for i in range(n_conv_layers):
        h = tf.keras.layers.Conv2D(filters, (3, 3), activation='relu', input_shape=(84, 84, 1), strides=strides[i])(h)

    input_to_fc = tf.keras.layers.Flatten()(h)
    y = input_to_fc
    for i in range(n_fc_layers - 1):
        y = tf.keras.layers.Dense(h_units, activation='relu')(y)

    output_layer = tf.keras.layers.Dense(output)(y)

    sr_nn = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    return sr_nn


def run():

    print(f"GPU is available: {tf.test.is_gpu_available()}")

    dataset = "omniglot"
    background_data, evaluation_data = load_omniglot(dataset, verbose=1)
    background_training_data, evaluation_training_data, evaluation_test_data = get_data_by_classes(background_data,
                                                                                                   evaluation_data)
    assert len(evaluation_training_data) == 659
    assert len(evaluation_test_data) == 659
    assert len(background_training_data) == 964

    assert evaluation_training_data[0].shape[0] == 15
    assert evaluation_test_data[0].shape[0] == 5
    assert background_training_data[0].shape[0] == 20

    s_learn, s_remember = partition_into_disjoint(background_training_data)

    sr_nn = sr_nn_model()
    print(sr_nn.summary())

    t = range(2000)
    tasks = None

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/classification/pretraining/omniglot/mrcl/gradient_tape/' + current_time + '/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    for epoch, v in enumerate(t):
        x_rand, y_rand = sample_random_10_classes(s_remember, background_training_data)
        x_traj, y_traj = sample_trajectory(s_learn, background_training_data)

        if epoch == 20:
            print("Changing the learning rate from 0.0001 to 0.00001")
            params["lr"] = 0.00001

        # TODO: Fix this
        """
        correct = 0

        for sample in x_rand:

            img = img.to(device)
            y = y.to(device)
            pred = maml(img)
            feature = F.relu(maml(img, feature=True))
            avg_feature = feature.mean(0)


            beta = args.beta
            beta_hat = avg_feature


            loss_rec = ((beta / (beta_hat+0.0001)) - torch.log(beta / (beta_hat+0.0001)) - 1)
            # loss_rec = (beta / (beta_hat)
            loss_rec = loss_rec * (beta_hat>beta).float()

            loss_sparse = loss_rec


            if args.l1:
                loss_sparse = feature.mean(0)
            loss_sparse = loss_sparse.mean()



            opt.zero_grad()
            loss = F.cross_entropy(pred, y)
            loss_sparse.backward(retain_graph=True)
            loss.backward()
            opt.step()
            correct += (pred.argmax(1) == y).sum().float()/ len(y)
            """
        loss = 0

        # Check metrics
        rep = sr_nn(x_rand)
        rep = np.array(rep)

        counts = np.isclose(rep, 0).sum(axis=1) / rep.shape[1]
        sparsity = np.mean(counts)

        with train_summary_writer.as_default():
            tf.summary.scalar('Sparsity', sparsity, step=epoch)
            tf.summary.scalar('Training loss', loss, step=epoch)
        print("Epoch:", epoch, "Sparsity:", sparsity, "Training loss:", loss.numpy())

        if epoch % 100 == 0:
            save_models(sr_nn, f"sr_nn_pretraining_mrcl_{epoch}_omniglot")


run()
