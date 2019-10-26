from omniglot_mrcl_evaluation import run as mrcl_omniglot_eval
import tensorflow as tf


def run_omniglot():
    """
    Learn MRCL online without any pre-training
    """

    results_folder = "scratch_omniglot_results"

    num_gb_to_use = 8

    gpus = tf.config.experimental.list_physical_devices('GPU')

    tf.config.experimental.set_virtual_device_configuration(gpus[0], [
        tf.config.experimental.VirtualDeviceConfiguration(memory_limit=(1024 * num_gb_to_use))])

    mrcl_omniglot_eval(results_folder=results_folder, load_saved=None)


def run_isw():
    """
    Learn SineWaves online without any pre-training
    """

    results_folder = "scratch_isw_results"

    # results = mrcl_eval()


run_omniglot()
