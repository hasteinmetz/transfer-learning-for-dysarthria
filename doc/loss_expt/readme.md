# Losses experiment

Compared the losses using mean CTC loss versus summed CTC loss to verify that L2Arctic (with longer utterances) was receiving too much weight and "distracting" the model from the goal.

The experiment was run on 10% of the dataset. The parameters were held the same.

See parent directory for more info. Debug config file used:


```
# This file contains a template that can be used to modify
# other wav2vec2.0 models. It contains 2 sections:
#   1. Trainer Arguments
#   2. Model Configuration
# The Trainer Arguments section is used to create 
# a HuggingFace TrainingArguments class to pass to the Trainer
# The Model Configuration section is used to add any additional
# arguments to the training script. You can also add other sections,
# and it can be accessed through wav2vec_args

[Paths]
data_dir: ~/thesis/data
model_name: debug
output_directory: models
result_directory: results

[Trainer Arguments]
output_dir = ${Paths:output_directory}/${Paths:model_name}
evaluation_strategy = epoch
learning_rate = 5e-4
train_batch_size = 3
eval_batch_size = 3
optim = adafactor
gradient_accumulation_steps = 400
seed = 2022
num_train_epochs = 1
eval_accumulation_steps = 1
remove_unused_columns = False
greater_is_better = False

# [Logging]
log_level = message
logging_strategy = epoch

# [Training/Eval Commands]
do_train = True
do_eval = True

# Use the following to resolve memory issues
full_determinism = False 
fp16 = False

[Training Configuration]
load_model = false
save_results = ${Paths:result_directory}/${Paths:model_name}
pretrained_model = facebook/wav2vec2-base-960h
model_type = multitask

# [Dataset Information]
dataset_path = ${Paths:data_dir}/dataset1
raw_data_path = ${Paths:data_dir}/processed_data
test_set = validation
generate_new_data = False
debug_dataset = True
filter_contols = True

# [Callbacks]
tensorboard = false
early_stopping = 0 # 0 for no stopping n > 0 for stopping

[Model Configuration]
dataset_loss_weighting = None
ctc_zero_infinity = True
```
