""" Loader file for different Tensorflow datasets"""

import tensorflow as tf
import tensorflow_datasets as tfds


def resize(image):
    """
    Resize an image to 84x84
    """
    image['image'] = tf.image.resize(image['image'], size=(84, 84))
    image['image'] = image['image'][:, :, 0]
    return image


def load_omniglot(transform=True, verbose=1):
    """
    Load Omniglot Dataset
    """
    # Load the data as defined in https://www.tensorflow.org/datasets/catalog/omniglot
    data, info = tfds.load(name="omniglot", with_info=True, shuffle_files=False)

    # Split into background (also named as pre-training) and evaluation
    background, evaluation = data['train'], data['test']

    # Resize images to 84x84
    if transform:
        background = background.map(lambda sample: resize(sample))
        evaluation = evaluation.map(lambda sample: resize(sample))

    if verbose > 0:
        print("Downloaded {} dataset (v:{})".format(info.name, str(info.version)))
        print("Size in disk: {} MB".format(info.size_in_bytes / 1e6))
    if verbose > 1:  # More info...
        print("Dataset info: \n", info.features)  # Inspect features
        print("Sample info: \n", background.element_spec)  # Inspect element

    return background, evaluation
