import pytest


def test_data_partitioner():
    import synth_datasets
    
    # Correct use case
    sine_data = synth_datasets.gen_sine_data(n_id=10, n_samples=320)
    partition = synth_datasets.partition_sine_data(sine_data, pretraining_n_seq=4, evaluation_n_seq=6, seq_len=320)
    assert "pretraining" in partition
    assert "evaluation" in partition
    pretraining = partition["pretraining"]
    evaluation = partition["evaluation"]
    assert len(pretraining["z"]) == len(pretraining["y"]) == len(pretraining["k"]) == 4 * 320
    assert len(evaluation["z"]) == len(evaluation["y"]) == len(evaluation["k"]) == 6 * 320
    
    # Wrong use case
    try:
        partition = synth_datasets.partition_sine_data(sine_data, pretraining_n_seq=4, evaluation_n_seq=5, seq_len=320)
    except Exception as e:
        assert type(e) == AttributeError
