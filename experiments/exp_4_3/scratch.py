from omniglot_mrcl_evaluation import run as mrcl_omniglot_eval


def run_omniglot():
    """
    Learn MRCL online without any pre-training
    """

    results_folder = "scratch_omniglot_results"

    mrcl_omniglot_eval(results_folder=results_folder, load_saved=None)


def run_isw():
    """
    Learn SineWaves online without any pre-training
    """

    results_folder = "scratch_isw_results"

    # results = mrcl_eval()


run_omniglot()
