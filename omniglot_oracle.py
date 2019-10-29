from omniglot_mrcl_pretraining import pretrain
from omniglot_mrcl_evaluation import evaluate

pretrain(sort_samples=False, model_name="oracle")
evaluate(sort_samples=False, model_name="pretraining_oracle_1900_omniglot.tf")  # TODO: check which epoch to load
