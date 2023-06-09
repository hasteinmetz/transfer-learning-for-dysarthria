#!/bin/env python

# packages:

from typing import *

from sys import exit

import os

from datasets import Audio
from datasets import disable_caching as hf_datasets_disable_caching

import torch
from transformers import Wav2Vec2Processor, EarlyStoppingCallback, TrainerCallback

# modules:

from eval import plot_all

from modeling import (BaseWav2Vec2, MultitaskWav2Vec2, OldMTWav2Vec2, AuxtaskWav2Vec2,
                      AuxWav2Vec2Config, Wav2VecTrainer, MultitaskWav2Vec2Config)
from modeling.multitask_aux import get_dann_task_values, TaskValues

from args import parse_arguments, Arguments

from logging_utils import set_up_logging, resolve_cuda, fix_batch_size

from dataset_utils import NewDatasetDict, stratified_slice

from process import get_preprocess_dataset_fn, filter_outliers

''' ############ Logging ############ '''
import logging
logger = logging.getLogger("wav2vec2_logger")
''' ################################# '''


''' ############ Set up Trainer callbacks ############ '''

def set_up_callbacks(args: Arguments) -> List[TrainerCallback]:
    '''Determine what callbacks to include in the trainer'''
    callbacks = []
    if 'early_stopping' in args.trial_config and args.trial_config['early_stopping'] != 0:
        es_callback = EarlyStoppingCallback(args.trial_config['early_stopping'])
        callbacks.append(es_callback)
        args.trainer_args.load_best_model_at_end = True
        args.trainer_args.save_total_limit = 1
        args.trainer_args.metric_for_best_model = 'wer'
    if 'tensorboard' in args.trial_config and args.trial_config['tensorboard']:
        from transformers.integrations import TensorBoardCallback
        tb_callback = TensorBoardCallback(args.trial_config['tensorboard'])
        callbacks.append(tb_callback)
    callbacks = callbacks if len(callbacks) > 1 else None
    return callbacks

''' ############ Load datasets & models ############ '''

def generate_audio(args: Arguments, dataset: NewDatasetDict, 
                       processor: Wav2Vec2Processor = None) -> NewDatasetDict:
    """Helper function to generate audio features of the dataset. (Separate function to disentangle
       the function from load_dataset and avoid growing cache)

    Args:
        args (Arguments): Training/model/scripting arguments
        dataset (NewDatasetDict): The dataset with file to generate audio features loaded in 
                                  load_dataset
        processor (Wav2Vec2Processor): Processor used to generate audio features

    Returns:
        NewDatasetDict: Final dataset with audio features
    """
    logger.message("Processing audio...")
    # preprocess the dataset
    def preprocess_labels(x):
        x['transcript'] = x['transcript'].strip().upper()
        x['transcript'] = x['transcript'].replace("-", " ")
        return x
    dataset = dataset.map(preprocess_labels, desc="Preprocessing text labels")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
    if processor is None:
        # Load new tokenizer to avoid caching bug: github.com/huggingface/datasets/issues/3847
        processor = Wav2Vec2Processor.from_pretrained(args.model_config['pretrained_model_name'])
    mapping_fn = get_preprocess_dataset_fn(processor)
    dataset = dataset.map(function=mapping_fn, desc="Generating audio features")
    return dataset


def configure_dataset(args: Arguments, dataset: NewDatasetDict, disable_caching: bool = True
                      ) -> Tuple[NewDatasetDict, Union[List[int], TaskValues]]:
    """Configure the dataset to the training paradigm by filtering or mapping dataset features
       disable_caching defaults to True to avoid large caching issues
    
    Args:
        args (Arguments): Training/model/scripting arguments
        dataset (NewDatasetDict): The dataset with file to generate audio features loaded in 
                                  load_dataset
        disable_caching (boolean | default=True): Whether to deleted the caches generated from 
                                                  the transformation processes at the end
                                                  
    Returns:
        Tuple of:
            - NewDatasetDict: Intermediate dataset with correct data for training
            - Union[List[int], TaskValues]: Task values to be used as inputs for the model
                                            TaskValues class if used by a DANN model
    """

    # filter out specific datasets (used to compare different finetuning model)
    if 'filter_datasets' in args.trial_config and args.trial_config['filter_datasets']:
        logger.message("Removing specified datasets...")
        # logger.warning(f"rows before filter: {dataset.num_rows, dataset['train'].unique('dataset')}")
        dataset = dataset.filter_datasets(args.trial_config['filter_datasets'])
        # logger.warning(f"rows after filter: {dataset.num_rows, dataset['train'].unique('dataset')}")
        
    if 'filter_controls' in args.trial_config and args.trial_config['filter_controls']:
        logger.message("Removing controls from UA-Speech and TORGO...")
        dataset = dataset.filter_controls()

    # disable caching after filtering for the most speed/memory compromise
    if disable_caching:
        hf_datasets_disable_caching()

    if 'no_control_layer' in args.model_config and args.model_config['no_control_layer']:
        logger.message("Merging tasks 0 and 2 (dysarthric and control)...")
        def merge_layers(x):
            x['task'] = 0 if x['task'] == 2 else x['task']
            return x
        dataset = dataset.map(merge_layers)
        print("merged:", dataset[list(dataset.keys())[0]]['task'][0:10])

    # downsample L2 speakers to specific ones
    if 'l2_speakers' in args.trial_config and args.trial_config['l2_speakers']:
        dataset = dataset.select_l2_speakers(args.trial_config['l2_speakers'])

    task_values = dataset.get_unique('task')

    # if auxiliary classifier, then generate a mapping from speakers to tasks
    if args.model_config['model_type'] == 'multitask_aux' and 'classes' in args.model_config:
        dataset, task_values = get_dann_task_values(args=args, task_values=task_values, 
                                                    dataset=dataset)

    return dataset, task_values


def load_dataset(args: Arguments, processor: Wav2Vec2Processor = None) -> NewDatasetDict:
    '''Load the dataset from json or cache
        - args: the arguments in the config file
        - processor: a Wav2Vec2Processor to get load the wav files and pad data
    '''
    logger.info("Loading dataset")
    
    # get path data
    new_data = args.trial_config['generate_new_data'] if 'generate_new_data' in args else False
    cache_dir = os.path.basename(args.trial_config['dataset_path'])
    if 'debug_dataset' in args and args.trial_config['debug_dataset']:
        cache_dir += "-debug"

    # load the dataset dict
    dataset = NewDatasetDict.load_dataset_dict(
        dataset_path=args.trial_config['dataset_path'],
        processed_data_path=args.trial_config['processed_data_path'],
        new=new_data,
        cache_dir=cache_dir
    )

    # filter outliers by audio length (TODO: REMOVE AND PLACE IN dataset_builder)
    speaker_label_map = lambda x: dataset.decode_class_label('speaker', x)
    logger.message("Filtering outliers...")
    dataset = filter_outliers(dataset, 0.95, speaker_label_map)

    # resolve the use of test vs. validation set
    # also, if just performing evaluations, then drop the training set
    if args.trainer_args.do_train:
        if args.trial_config['test_set'].startswith('val'):
            dataset['test'] = dataset.pop('validation')
        else:
            logger.watch_out("Training a model using the test set is not recommended for a scientific study!")
            dataset.pop('validation')
            if 'debug_dataset' in args.trial_config and args.trial_config['debug_dataset']:
                logger.critical("Watch out! You're debugging with the test data! Fix the config file.")
                exit()
    else:
        dataset.pop('train')
        if args.trial_config['test_set'].startswith('val'):
            dataset['test'] = dataset.pop('validation')
        elif args.trial_config['test_set'].startswith('test'):
            dataset.pop('validation')
        elif args.trial_config['test_set'] == 'all' or args.trial_config['test_set'] == 'both':
            logger.message("Keeping both validation and test sets")
        else:
            raise ValueError("'test_set' should be one of: 'test', 'validation', or 'all'/'both'")
    
    # resolve the use of a debug dataset
    if 'debug_dataset' in args.trial_config and args.trial_config['debug_dataset']:
        logger.message("Stratifying debug dataset...")
        dataset = stratified_slice(dataset, 1.25)

    # processes text and generate audio data (see generate_audio function above)
    # generate audio first to minimize caching (since this data is shared across configurations)
    dataset = generate_audio(args, dataset, processor)

    # configure data to conform with training paradigm (see configure_dataset function above)
    dataset, task_values = configure_dataset(args, dataset, disable_caching=True) # disable_caching deletes caches upon completion

    return dataset, task_values


def load_model(
        args: Arguments, 
        task_values: Optional[List[int]] = None, 
        **kwargs
    ) -> Union[MultitaskWav2Vec2, BaseWav2Vec2, AuxtaskWav2Vec2]:
    '''Load the multitask or base model from config file arguments'''

    if 'model_type' not in args.model_config:
        raise ValueError("\"model_type\" is a required parameter in the config file." 
                         "Must be \"base\" or \"multitask\".")
    model_type = args.model_config['model_type']

    if model_type == 'multitask':
        model_class, config_class = MultitaskWav2Vec2, MultitaskWav2Vec2Config
        args.model_config['task_values'] = task_values
    elif model_type == 'multitask_aux' or 'aux' in model_type:
        model_class, config_class = AuxtaskWav2Vec2, AuxWav2Vec2Config
        args.model_config['task_values'] = task_values.tasks
        args.model_config['task_mapping'] = task_values.mapping
    elif model_type == 'base' or 'wav2vec' in model_type:
        model_class, config_class = BaseWav2Vec2, MultitaskWav2Vec2Config
        args.model_config['task_values'] = task_values
    else:
        raise ValueError(f"model needs to be \"base\" or \"multitask\"." f"Received {model_type}")

    load_model = 'load_model' in args.trial_config and args.trial_config['load_model']
    resume_checkpoint = 'resume_from_checkpoint' in args and args.trainer_args.resume_from_checkpoint
    if load_model or resume_checkpoint:
        if resume_checkpoint: # load checkpoint
            logger.warning("Resuming from checkpoint. " + 
                           "If you want to train new model, alter the .conf file.")
            assert args.trainer_args.ddp_find_unused_parameters == True, \
                   "ddp_find_unused_parameters must be true"
            checkpoints = sorted([d for d in os.listdir() if 'checkpoint' in d])
            output_dir = checkpoints[-1] # the most recent checkpoint
        else:
            output_dir = args.trainer_args.output_dir
        logger.warning(f"Loading model from {output_dir} (type: {model_type})")
        config = config_class.from_pretrained(pretrained_model_name_or_path=output_dir)
        # if it's an old multitask model, then use the oldmtwav2vec class
        if not hasattr(config, 'reinitialize_last_n_layers'):
            model_class = OldMTWav2Vec2
        model = model_class.from_pretrained(pretrained_model_name_or_path=output_dir, 
                                            config=config, **kwargs)
    else:
        if 'load_model' not in args.trial_config or not args.trial_config['load_model']:
            logger.warning(f"Training new model (type: {model_type}). " 
                           f"Will save at {args.trainer_args.output_dir}")
        config = config_class(**args.model_config)
        model = model_class.from_pretrained(
            pretrained_model_name_or_path=args.model_config['pretrained_model_name'],
            config=config,
            **kwargs
        )

    return model


''' ############ Trainer ############ '''

def set_up_trainer(
        model: Union[MultitaskWav2Vec2, BaseWav2Vec2, AuxtaskWav2Vec2],
        processor: Wav2Vec2Processor,
        args: Arguments,
        dataset: NewDatasetDict
    ) -> Wav2VecTrainer:
    '''Create a trainer to use for training the model'''
    callbacks = set_up_callbacks(args)
    classlabels = dataset.info.features
    trainer = Wav2VecTrainer(
        model=model,
        processor=processor,
        callbacks=callbacks,
        features=classlabels,
        train_dataset=dataset['train'] if 'train' in dataset else None,
        eval_dataset=dataset['test'],
        args=args
    )
    return trainer


''' ############ Train ############ '''

def main():
    '''Main driver to train a model'''

    # Load configuration and datasets
        
    arguments = parse_arguments()
    logger = set_up_logging(args=arguments)

    torch.manual_seed(arguments.trainer_args.seed)

    device, device_names = resolve_cuda(arguments.cmdline_args.cpu)
    
    # fix batch sizes when the QUATRO GPU is unavailable
    batch_size = arguments.trainer_args.train_batch_size
    if any(['Tesla' in dname for dname in device_names]) and batch_size > 4:
        arguments.trainer_args = fix_batch_size(arguments.trainer_args)

    if device == 'mps' and not arguments.cmdline_args.cpu:
        logger.message("Using mps for training.")
        arguments.trainer_args.use_mps_device = True

    if hasattr(arguments.cmdline_args, 'local_rank'):
        local_rank = arguments.cmdline_args.local_rank

    processor = Wav2Vec2Processor.from_pretrained(arguments.model_config['pretrained_model_name'])
    dataset, task_values = load_dataset(arguments)
    model = load_model(arguments, task_values)
    
    trainer = set_up_trainer(
        model=model,
        processor=processor,
        args=arguments,
        dataset=dataset
    )
    
    logger.message("Loaded configurations!")

    # TRAIN

    if arguments.trainer_args.do_train:
        logger.message("Training model...")
        trainer.train()
        logger.message("Saving model...")
        trainer.save_model()
        trainer.save_state()

    # EVAL
    if arguments.trainer_args.do_eval:
        output_dir = os.path.join(arguments.trainer_args.output_dir, 'trainer_state.json')
        if os.path.exists(output_dir):
            plot_all(output_dir)
        results_directory = arguments['save_results'] if 'save_results' in arguments else None
        test_set = arguments['test_set'] if 'test_set' in arguments else 'validation'
        
        def make_predictions():
            '''Driver to make predictions'''
            save_location = os.path.join(results_directory, test_set)
            logger.message(f"Evaluating model on {test_set}...")
            if not os.path.exists(save_location):
                os.makedirs(save_location)
            trainer.detailed_predict(
                save_dir = save_location
            )
        
        if test_set == 'all' or test_set == 'both':
            for test_set in ['validation', 'test']:
                trainer.eval_dataset = dataset[test_set]
                make_predictions()
        else:
            make_predictions()


if __name__ == '__main__':
    main()
