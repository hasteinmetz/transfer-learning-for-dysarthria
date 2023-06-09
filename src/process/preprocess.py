#!/bin/env python

'''This file contains function used to process datasets'''

from transformers import Wav2Vec2Processor
from datasets import (
    Dataset,
    DatasetDict
)
from typing import *

''' ############ Logging ############ '''
import logging
logger = logging.getLogger("wav2vec2_logger")
''' ################################# '''


Example = Dict[Any, Any]

# NOTE: TRAINER SHUFFLES AT EACH EPOCH


def preprocess_batch(    
        batch: Union[Dataset, DatasetDict], 
        processor: Wav2Vec2Processor
    ):
    # Featurizer
    batch['input_values'] = processor(
        audio=[x['array'] for x in batch['audio']],
        sampling_rate=16_000,
        return_tensors=None
    ).input_values
    # Lower case
    for i, tr in enumerate(batch["transcript"]):
        batch["transcript"][i] = tr.upper()
    # Input IDs
    batch["labels"] = [
        t for t in processor(
            text=batch["transcript"], return_tensors=None
        ).input_ids
    ]
    # TODO: Add task here
    return batch


def get_preprocess_dataset_fn(
        processor: Wav2Vec2Processor
    ):
    '''Take a batch of raw data and transform for the model
            Code adapated from tutorial found at: https://huggingface.co/docs/transformers/tasks/asr
    '''
    
    def map_processor(ex: Example, processor: Wav2Vec2Processor):
        audio = ex['audio']
        text = ex['transcript'].upper()
        inputs = processor(
            audio=audio['array'],
            sampling_rate=audio['sampling_rate'],
        )
        inputs['input_values'] = inputs['input_values'][0]
        inputs['labels'] = processor.tokenizer(text).input_ids
        inputs["input_length"] = len(inputs["input_values"])
        inputs['task'] = ex['task']
        return inputs

    return lambda batch: map_processor(batch, processor)