import torch
from transformers import TrainingArguments
import os
from math import floor

''' 
This module contains functions that help log information on the CUDA environment
of the currently running job
'''

''' ############ Logging ############ '''
import logging
logger = logging.getLogger("wav2vec2_logger")
''' ################################# '''

def query_cuda_memory():
    d_count = torch.cuda.device_count()
    memory_total = [
        torch.cuda.get_device_properties(i).total_memory for i in range(d_count)
    ]
    memory_available = [
        memory_total[i] - (torch.cuda.memory_reserved(i) + torch.cuda.memory_allocated(i)) for i in range(d_count)
    ]
    logger.info(f"CUDA memory total: {memory_total}")
    logger.info(f"Environment: CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    logger.info(f"CUDA memory available: {memory_available}")


def resolve_cuda(cpu_allowed: bool = False):
    if torch.cuda.is_available():
        device = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        d_count = torch.cuda.device_count()
        device_name = [torch.cuda.get_device_name(i) for i in range(d_count)]
        logger.message(f"Using device(s): {device}. Device name: {'|'.join(device_name)}")
        query_cuda_memory()
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        device_name = ['mps']
    else:
        device = 'cpu'
        device_name = ["cpu"]
    if device == 'cpu' and not(cpu_allowed):
        logger.warning(
            "Using cpu. Exiting... If Condor submit file is correctly configured, the job may restart"
        )
        from datetime import datetime
        with open("results/logs/HTCondor.log", 'a') as log:
            print(
                f"{datetime.now()} Trying to restart job...", file=log
            )
        exit(130)

    return device, device_name


def fix_batch_size(args: TrainingArguments) -> TrainingArguments:
    """Fixes values in training arguments if the batch size is too large

    Args:
        args (TrainingArguments): TrainingArguments class

    Returns:
        TrainingArguments: modified TrainingArguments class
    """
    logger.message("Setting batch size to 4")
    original_batch_size = args.per_device_train_batch_size
    args.per_device_train_batch_size = 2
    args.per_device_eval_batch_size = 2
    args.gradient_accumulation_steps = floor(original_batch_size/2)
    return args


if __name__ == '__main__':
    pass