#!/bin/env python

'''
This script is used to evaluate the performance the base wav2vec2-960h model on 
different kinds of modified speech
'''

import os, json
import numpy as np
import pandas as pd
from datasets import Dataset, Audio
from transformers import (Wav2Vec2ForCTC, Wav2Vec2Processor, 
                          Trainer, TrainingArguments, HfArgumentParser)
from dataset_creation.modded_speech_data import (create_modded_speech_dataset,
                                                 DEFAULT_CTL_LIST, DEFAULT_DYS_LIST, 
                                                 DEFAULT_MAX_DURATION)
import torch
from torch import Tensor
from dataclasses import dataclass
from eval import compute_wer, INTELLIGIBILITY_DATA
from process import get_preprocess_dataset_fn
from argparse import ArgumentParser
from typing import *

INTL = INTELLIGIBILITY_DATA['UASpeech']

@dataclass
class CTCCollator:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = "longest"
    return_demographics: bool = False
    return_task: bool = False

    def __call__(self, batch: List[Dict[str, Union[List[int], Tensor]]]
                 ) -> Dict[str, Tensor]:
        '''Take a batch input, process the input (pad it), and return a 
        valid input to the model.'''
        # prep labels and inputs
        input_values = [{"input_values": b["input_values"]} for b in batch]
        label_ids = [{"input_ids": b["labels"]} for b in batch]

        # pad inputs
        inputs = self.processor.feature_extractor.pad(
            processed_features=input_values,
            padding=self.padding,
            pad_to_multiple_of=8,
            return_tensors='pt'
        )

        # pad labels
        labels = self.processor.tokenizer.pad(
            encoded_inputs=label_ids, 
            padding=self.padding,
            pad_to_multiple_of=8,
            return_tensors='pt'
        )

        inputs['labels'] = labels['input_ids'].masked_fill(labels.attention_mask.ne(1), -100)
        
        return inputs


def load_model(dataset: Dataset, args: TrainingArguments, processor: Wav2Vec2Processor,
               device: str = 'cpu', pretrained_model: str = 'facebook/wav2vec2-base-960h'
               ) -> Trainer:
    '''Load the model, tokenizer and return a trainer class'''
    model = Wav2Vec2ForCTC.from_pretrained(pretrained_model)
    model = model.to(device)
    collator = CTCCollator(processor)
    wer = compute_wer(processor)
    trainer = Trainer(model, args, collator, eval_dataset=dataset, compute_metrics=wer)
    return trainer


def argumentparse():
    arguments = ArgumentParser()
    arguments.add_argument('--dataset', '-d', type=str, required=True,
                           help='Path to dataset file')
    arguments.add_argument('--model', '-m', type=str, required=False,
                           help='Path to model to evaluate')
    arguments.add_argument('--new_data', '-new', action='store_true',
                           help='Whether or not to generate new baseline dataset')
    args, remaining_args = arguments.parse_known_args()
    return args, remaining_args


def load_data(args: ArgumentParser):
    if not os.path.exists(args.dataset) or args.new_data:
        if os.path.exists(args.dataset):
            print("Found previous dataset generation. Removing previous cache...")
            dataset = Dataset.load_from_disk(args.dataset)
            dataset.cleanup_cache_files()
        else:
            print(f"{args.dataset} not found. Generating new data using " 
                  "'data/uaspeech/metadata.csv' to processed_data/baselines/...")
        commondir = os.path.commonpath([__file__, os.path.abspath(args.dataset)]) 
        fpath = lambda x: os.path.join(commondir, x)
        default_mods = DEFAULT_CTL_LIST if "ctl" in args.dataset else DEFAULT_DYS_LIST
        create_modded_speech_dataset(fpath('data/uaspeech/data/audio/modified'), False, 
                                     fpath('data/uaspeech/metadata.csv'), args.dataset,
                                     default_mods, DEFAULT_MAX_DURATION)
    
    return Dataset.load_from_disk(args.dataset)


def prepare_data(dataset: Dataset, processor: Wav2Vec2Processor) -> Dataset:
    '''Read the dataset and process the audio for the model'''
    def preprocess_labels(x):
        x['transcript'] = x['transcript'].strip().upper()
        x['transcript'] = x['transcript'].replace("-", " ")
        return x
    dataset = dataset.map(preprocess_labels, desc="Preprocessing text labels")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
    mapping_fn = get_preprocess_dataset_fn(processor)
    dataset = dataset.map(function=mapping_fn, desc="Generating audio features")
    return dataset


def main():
    args, _ = argumentparse()
    
    dataset = load_data(args)

    hfparser = HfArgumentParser(TrainingArguments)
    training_args, _ = hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    training_args.per_device_eval_batch_size = 4
    training_args.eval_accumulation_steps = 4
    processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
    dataset = prepare_data(dataset, processor)
    trainer = load_model(dataset, training_args, processor, device)

    pred_logits = trainer.predict(trainer.eval_dataset).predictions
    pred_argmax = np.argmax(pred_logits, axis=-1)
    predictions = processor.batch_decode(pred_argmax)
    pred_df = dataset.add_column('prediction', predictions).to_pandas()
    pred_df = pred_df[['prediction', 'transcript', 'mod', 'factor', 'speaker', 'id', 'gender']]
    pred_df['intl'] = pred_df['speaker'].map(lambda ex: INTL["U" + ex])
    pred_df.to_csv('results/baselines/predictions_dys.csv')

    wer = trainer.compute_metrics

    pred_df['condition'] = pred_df['mod'].str.cat(pred_df['factor'], sep="_")
    conditions = pred_df['condition'].unique()
    wer_by_condition = {}
    for c in conditions:
        tmp_df = pred_df[pred_df['condition']==c]
        wer_by_condition[c] = wer.compute(predictions=tmp_df['prediction'].fillna("").to_list(),
                                          labels=tmp_df['transcript'].to_list())
    print(wer_by_condition)
    with open('results/baselines/baseline_dys.json', 'w') as baselines:
        json.dump(wer_by_condition, baselines, indent=4)
    
    # compute wer by intelligibility
    metrics = lambda x: wer.compute(predictions=x['prediction'].fillna("").to_list(),
                                    labels=x['transcript'].to_list())
    agg = (pred_df.groupby(['intl', 'speaker', 'condition'])
                  .apply(metrics)
                  .apply(pd.Series)
                  .reset_index(level='condition')
                  .pivot(columns='condition'))
    index_key = {'very low': 0, 'low': 1, 'mid': 2, 'high': 3}
    index_value = {v: k for k, v in index_key.items()}
    agg.index = agg.index.map(lambda x: (index_key[x[0]], x[1]))
    agg = agg.sort_index(level=0)
    agg.index = agg.index.map(lambda x: (index_value[x[0]], x[1]))
    agg = agg.round(3) * 100
    print(agg)
    agg.to_csv('results/baselines/dys_by_intl.csv')


if __name__ == '__main__':
    main()
