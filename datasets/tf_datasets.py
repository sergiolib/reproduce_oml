""" Loader file for different Tensorflow datasets"""

import tensorflow_datasets as tfds


def load_data(name, verbose=1):
    data, info = tfds.load(name=name, with_info=True, shuffle_files=False)  # We don't want shuffled files right?
    train_data, test_data = data['train'], data['test']

    if verbose > 0:
        print("Downloaded {} dataset (v:{})".format(info.name, str(info.version)))
        print("Size in disk: {} MB".format(info.size_in_bytes / 1e6))
    if verbose > 1:  # More info...
        print("Dataset info: \n", info.features)  # Inspect features
        print("Sample info: \n", train_data.element_spec)  # Inspect element

    return train_data, test_data


