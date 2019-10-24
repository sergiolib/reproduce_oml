def run_isw():
    """
    Learn MRCL online without any pre-training

    TODO: Does from scratch mean when online training, update the whole network or just as MRCL, update only TLN?

    """

    # make sure network starts with a random initialization
    results = mrcl_evaluate(eval_data, rln, tln, regression_parameters)
    np.savetxt("scratch_eval_results.txt", results)
    np.savetxt("scratch_ground_truth.txt", np.array(eval_data["y"]))


def run_omniglot():
    pass