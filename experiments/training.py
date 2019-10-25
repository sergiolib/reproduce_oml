import tensorflow as tf

from os import makedirs
from os.path import isdir


def copy_parameters(source, dest):
    for s, d in zip(source.trainable_variables, dest.trainable_variables):
        d.assign(s)


def save_models(epoch, rln, tln):
    try:
        isdir("saved_models/")
    except NotADirectoryError:
        makedirs("save_models")
    rln.save(f"saved_models/rln_pretraining_{epoch}.tf", save_format="tf")
    tln.save(f"saved_models/tln_pretraining_{epoch}.tf", save_format="tf")


@tf.function
def inner_update(x, y, tln, rln, beta, loss_fun):
    with tf.GradientTape(watch_accessed_variables=False) as Wj_Tape:
        Wj_Tape.watch(tln.trainable_variables)
        inner_loss = compute_loss(x, y, tln=tln, rln=rln, loss_fun=loss_fun)
    gradients = Wj_Tape.gradient(inner_loss, tln.trainable_variables)
    for g, v in zip(gradients, tln.trainable_variables):
        v.assign_sub(beta * g)


@tf.function
def compute_loss(x, y, tln, rln, loss_fun):
    output = tln(rln(x))
    loss = loss_fun(output, y)
    return loss


def pretrain_mrcl(x_traj, y_traj, x_rand, y_rand, tln, tln_initial, rln, meta_optimizer, loss_function, beta):
    # Random reinitialization of last layer
    w = tln.layers[-1].weights[0]
    b = tln.layers[-1].weights[1]
    new_w = tln.layers[-1].kernel_initializer(shape=w.shape)
    new_b = tf.keras.initializers.zeros(shape=b.shape)
    w.assign(new_w)
    b.assign(new_b)

    copy_parameters(tln, tln_initial)

    # Sample x_rand, y_rand from s_remember
    x_shape = x_traj.shape
    x_traj_f = tf.reshape(x_traj, (x_shape[0] * x_shape[1], x_shape[2]))
    y_traj_f = tf.reshape(y_traj, (x_shape[0] * x_shape[1],))

    x_meta = tf.concat([x_rand, x_traj_f], axis=0)
    y_meta = tf.concat([y_rand, y_traj_f], axis=0)

    for x, y in tf.data.Dataset.from_tensor_slices((x_traj, y_traj)):
        inner_update(x=x, y=y, tln=tln, rln=rln, beta=beta, loss_fun=loss_function)

    with tf.GradientTape(persistent=True) as theta_Tape:
        outer_loss = compute_loss(x=x_meta, y=y_meta, tln=tln, rln=rln, loss_fun=loss_function)

    tln_gradients = theta_Tape.gradient(outer_loss, tln.trainable_variables)
    rln_gradients = theta_Tape.gradient(outer_loss, rln.trainable_variables)
    del theta_Tape
    meta_optimizer.apply_gradients(zip(tln_gradients + rln_gradients,
                                       tln_initial.trainable_variables + rln.trainable_variables))

    copy_parameters(tln_initial, tln)

    return outer_loss


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


def basic_pt_eval(data, model, params, layers_to_freeze=1):
    batch_size = params["random_batch_size"]
    optimizer = params["optimizer"](learning_rate=params["learning_rate"])
    results = []

    # TODO: not sure if this works!!! this is probably deprecated in tf 2.0
    # create a var list of layers you want to train as trainable variables
    for layer_i in range(layers_to_freeze):
        model.get_layer("layer", layer_i).trainable = False

    for i in range(0, len(data["x"]), batch_size):
        batch = {a: data[a][i:i + batch_size] for a in data}

        with tf.GradientTape() as tape:
            outputs = model(batch["x"])
            loss = params["loss_metric"](outputs, batch["y"])

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        results += [np.array(outputs)]
    return np.concatenate(results)
