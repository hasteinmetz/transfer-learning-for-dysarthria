#!/bin/env python

from torch.utils.data import DataLoader
import torch
from datasets import Dataset, ClassLabel
from dataclasses import dataclass
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    Trainer
)
from random import choices as randomly_sample
from .multitask import MultitaskWav2Vec2
from eval import (compute_wer, get_detailed_descriptives, uaspeech_unseen, 
                  add_hyperparameters_to_results)
import numpy as np
import os
from typing import *


''' ############ Logging ############ '''
import logging
logger = logging.getLogger("wav2vec2_logger")
''' ################################# '''


def collate_fn(data: Dataset) -> Dict[str, Any]:
    '''Collate elements of a mini-batch to form a tensor (for dataloader)'''
    batch = {}
    batch['input_values'] = torch.stack(
        [torch.tensor(t['input_values']) for t in data], dim=0
    )
    batch['labels'] = torch.stack(
        [torch.tensor(t['labels']) for t in data], dim=0
    )
    batch['transcript'] = [t['transcript'] for t in data]
    batch['speaker'] = [t['speaker'] for t in data]
    return batch


class CTCDataloader(DataLoader):
    def __init__(self, dataset: Union[Dataset, torch.utils.data.Dataset], **kwargs):
        '''Create a torch.utils.data.DataLoader class to collate the data correctly'''
        if 'dataset' in kwargs:
            kwargs.pop('dataset')
        if 'collate_fn' in kwargs:
            logger.warning(
                "Collation function is built into CTCDataloader class. Ignoring argument."
            )
            kwargs.pop('collate_fn')
        super().__init__(
            dataset=dataset,
            collate_fn=collate_fn,
            **kwargs
        )


@dataclass
class CTCCollator:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = "longest"
    return_demographics: bool = False
    return_task: bool = False

    def __call__(
            self, 
            batch: List[Dict[str, Union[List[int], torch.Tensor]]]
        ) -> Dict[str, torch.Tensor]:
        '''Take a batch input, process the input (pad it), and return a 
        valid input to the model.
        Code adapated from tutorial found at: https://huggingface.co/docs/transformers/tasks/asr'''
        # Separate demographics data
        demographics = ['transcript', 'speaker', 'l1', 'gender']
        demographics = [{k: b[k] for k in demographics} for b in batch]

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

        inputs['labels'] = labels['input_ids'].masked_fill(
            labels.attention_mask.ne(1), -100
        )
        
        if self.return_task:
            tasks = [b["task"] for b in batch]
            inputs['task'] = torch.tensor(
                tasks, 
                dtype=torch.uint8,
                requires_grad=False
            )

        if self.return_demographics:
            return inputs, demographics
        else:
            return inputs

''' ######################## Trainer ######################## '''

class Wav2VecTrainer(Trainer):
    def __init__(
            self,
            model: Union[MultitaskWav2Vec2, Wav2Vec2ForCTC],
            processor: Wav2Vec2Processor,
            args: "args.Arguments",
            features: Dict[str, ClassLabel],
            **kwargs
        ) -> None:
        '''Subclass of Trainer which tests for the fixed data collator and metrics'''

        self.processor = processor

        if 'compute_metrics' in kwargs:
            logger.warning("""
            (Wav2VecTrainer.__init__) compute_metrics argument provided. Removing this argument and using WER.
            To enter custom compute_metrics arguments, edit the Wav2VecTrainer class in train_utils.py.
            """)
            kwargs.pop('compute_metrics')

        self.wer = compute_wer(self.processor)

        if 'data_collator' in kwargs:
            logger.warning("""
            (Wav2VecTrainer.__init__) data_collator argument provided. Removing this argument collator 
            defined intrain_utils.py. To modify the collator edit the CTCCollator class in train_utils.py.
            """)
            kwargs.pop('data_collator')

        if not hasattr(args.trainer_args, 'remove_unused_columns'):
            if args.trainer_argsremove_unused_columns:
                logger.warning("""
                (Wav2VecTrainer.__init__) providing `remove_unused_columns = True` as an argument
                may result in the CTCCollator breaking. The current implementation calculates values
                on the fly, but you may wish to preprocess the data
                """)

        self.classlabels = features

        self.mode = 'none'

        self.model_type = type(model).__name__

        if 'eval_dataset' in kwargs:
            self.subtasks = kwargs['eval_dataset'].unique('speaker_type')
        else:
            self.subtasks = None

        collator = CTCCollator(processor, return_task=True)
        
        super().__init__(
            model=model,
            compute_metrics=self.wer,
            data_collator=collator,
            args=args.trainer_args,
            **kwargs
        )

    '''##################### Modify or freeze model utilities'''

    def freeze_encoder(self):
        '''Freezes the encoder of the base wav2vec2 model'''
        for param in self.model.wav2vec2.encoder.parameters():
            param.requires_grad = False
        self.model.wav2vec2.encoder.requires_grad = False

    
    def freeze_extractor(self):
        '''Freezes the encoder of the base wav2vec2 model'''
        self.model.wav2vec2.freeze_feature_encoder()

    
    def reinitialize_layers(self, reinit_layers: int):
        '''Reinitialize the last layers, as detailed in Pasad et al. (2021)'''
        if isinstance(self, Wav2Vec2ForCTC) and not isinstance(self, MultitaskWav2Vec2):
            last_three_layers = self.model.wav2vec2.encoder.layers[-reinit_layers:]
            init_weights = lambda module: self.model._init_weights(module)
            for layer in last_three_layers:
                layer.apply(init_weights)

    '''##################### Training, eval, and predict'''

    def trainer_headings(self) -> None:
        '''Log information on training
            Code adapted from: https://github.com/huggingface/transformers/blob/v4.24.0/src/transformers/trainer.py#L209
        '''

        args = self.args
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size
        num_examples = self.train_dataset.num_rows
        num_update_steps_per_epoch = max(num_examples // args.gradient_accumulation_steps, 1)
        max_steps = args.num_train_epochs * num_update_steps_per_epoch
        
        logger.message("***** Running training *****")
        logger.message(f"  Num examples = {num_examples}")
        logger.message(f"  Num Epochs = {args.num_train_epochs}")
        logger.message(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.message(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.message(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        logger.message(f"  Total optimization steps = {max_steps}")
        logger.message(
            f"  Number of trainable parameters = {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}"
        )


    def train(self, resume_from_checkpoint: Optional[Union[str, bool]] = None, trial: Union["optuna.Trial", Dict[str, Any]] = None, ignore_keys_for_eval: Optional[List[str]] = None, **kwargs):
        if hasattr(self.model.config, 'freeze_feature_extractor') and self.model.config.freeze_feature_extractor:
            logger.warning("Freezing feature extractor for training")
            self.freeze_extractor()
        if hasattr(self.model.config, 'freeze_encoder') and self.model.config.freeze_encoder:
            logger.warning("Freezing encoder layers for training")
            self.freeze_encoder()
        if hasattr(self.model.config, 'reinitialize_last_n_layers') and self.model.config.reinitialize_last_n_layers:
            logger.warning("Reinitializing the encoder's last " + 
                           f"{self.model.config.reinitialize_last_n_layers} layers for training")
            self.reinitialize_layers(self.model.config.reinitialize_last_n_layers)
        self.trainer_headings()
        self.mode = 'train'
        out = super().train(resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)
        self.mode = 'eval'
        return out


    def decode_batch_features(self, features: List[int], batch: List[Dict[Any, Any]]) -> Tuple[List[str]]:
        '''Return a list of strings corresponding to a list of features to decode'''
        decoded = []
        for f in features:
            decoded.append([self.classlabels[f].int2str(x[f]) for x in batch])
        return tuple(decoded)
    

    def evaluate(
            self, 
            eval_dataset: Optional[Dataset] = None, 
            ignore_keys: Optional[List[str]] = None, 
            metric_key_prefix: str = "eval"
        ) -> Dict[str, float]:
        '''Override evaluate method to allow for task specific metrics logging
           This method performs at most three eval loops to get metrics for each subtask
           Code adapted from: github.com/huggingface/transformers/
        '''
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        ignore_keys = ['hidden_states', 'attentions', 'logits_classifier'] if ignore_keys is None else ignore_keys

        output = self.predict(eval_dataset, ignore_keys, metric_key_prefix)

        # CUSTOM BEHAVIOR BEGINS HERE:

        pred_argmax = np.argmax(output.predictions, axis=-1)
        pred_argmax[pred_argmax == -100] = self.processor.tokenizer.pad_token_id
        
        if self.subtasks is None:
            self.subtasks = eval_dataset.unique('speaker_type')
        for subtask in self.subtasks:
            indices = [i for i, v in enumerate(eval_dataset) if v['speaker_type'] == subtask]
            preds_num = pred_argmax[indices]
            labels_caps = eval_dataset[indices]['transcript']
            preds_caps = self.processor.batch_decode(preds_num)
            subtask_p, subtask_l = [p.lower() for p in preds_caps], [l.lower() for l in labels_caps]
            sub_outputs = self.wer.compute(predictions=subtask_p, labels=subtask_l)
            prefix = subtask + "_" + metric_key_prefix
            sub_outputs = {prefix + "_" + k: v for k, v in sub_outputs.items()}
            output.metrics.update(sub_outputs)

        # CUSTOM BEHAVIOR ENDS HERE

        self.log(output.metrics)

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, output.metrics)

        return output.metrics


    def get_eval_dataloader(self, 
            eval_dataset: Optional[Dataset] = None, 
            return_demographics: Optional[bool] = False
        ) -> DataLoader:
        '''Get the eval Dataloader with or without a wrapper to get demographics batch data'''
        if return_demographics:
            self.data_collator.return_demographics = True
        if self.model_type == 'MultitaskWav2Vec2':
            self.data_collator.return_task = True
        return super().get_eval_dataloader(eval_dataset)


    def detailed_predict(
            self, 
            dataset: Optional[Dataset] = None, 
            ignore_keys: Optional[List[str]] = None, 
            save_dir: Optional[str] = None
        ) -> Dict[str, float]:
        '''Obtain detailed the model output information and save
        this output info to a csv file for future analysis'''

        ignore_keys = ['hidden_states', 'attentions', 'logits_classifier'] if ignore_keys is None else ignore_keys

        ds = dataset if dataset is not None else self.eval_dataset

        pred_logits = self.predict(test_dataset=ds, ignore_keys=ignore_keys, 
                                   metric_key_prefix="detailed_pred").predictions
        pred_argmax = np.argmax(pred_logits, axis=-1)
        predictions = self.processor.batch_decode(pred_argmax)

        pred_df = (ds
            .add_column('prediction', predictions)
            .remove_columns(["audio", "input_values", "input_length", "labels"])
            .to_pandas()
        )

        pred_df['transcript'] = pred_df['transcript'].map(lambda x: x.upper())

        for feature in ['speaker', 'gender', 'l1']:
            pred_df[feature] = pred_df[feature].map(
                lambda x: self.classlabels[feature].int2str(x)
            )
        
        if save_dir:
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
            save_file = os.path.join(save_dir, "results.csv")
            logger.info(f"Saving predictions to {save_file}")
            pred_df.to_csv(save_file)

            get_detailed_descriptives(pred_df, save_dir)
            uaspeech_unseen(pred_df, '~/thesis/data/uaspeech/metadata.csv', save_dir) # metadata directory hard coded for now
            add_hyperparameters_to_results(self.args.output_dir, save_dir)

        logger.message("Done evaluation!")

        return pred_df


if __name__ == '__main__':
    pass