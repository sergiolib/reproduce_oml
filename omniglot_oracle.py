from omniglot_mrcl_pretraining import pretrain
from omniglot_mrcl_evaluation import evaluate

pretrain(sort_samples=False, model_name="oracle")
evaluate("pretraining_oracle_14999_omniglot.tf")
