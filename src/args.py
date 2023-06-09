#!/bin/env python

'''File to parse arguments for train.sh'''

import argparse
import configparser as configparse
from configparser import ExtendedInterpolation
import re
from os.path import realpath, expanduser
import os
import json
from transformers import TrainingArguments
from dataclasses import dataclass
import logging
from typing import *
import torch


''' ############ Logging ############ '''
import logging
logger = logging.getLogger("wav2vec2_logger")
''' ################################# '''


class ConfigSyntaxError(SyntaxError):
    pass


class PathInterpolation(ExtendedInterpolation):
    '''Resolve relative links'''
    def before_get(self, parser, section: str, option: str, value: str, defaults) -> str:
        value = super().before_get(parser, section, option, value, defaults)
        if re.match(r'^(\.\./)+', value):
            value = str(realpath(value))
        if re.match(r'^~', value):
            value = re.sub('^~', str(expanduser("~")), value)
        return value


def parse_dict_types(d: Dict[str, str]) -> Dict[str, Any]:
    trainer_args = {}
    for key, value in d.items():
        value = re.sub(r'(^\s+|\s+$)', '', value) # remove unwanted spacing
        # lists
        if re.match(r'^.*,.*$', value):
            _value = str(value).split(",")
            value = [v.strip() for v in _value]
            # if it's a string of numbers (only for floats)
            if all([(r'^-?(\d+\.\d*|\d*\.\d+)$', v) is not None for v in value]):
                value = [float(v) for v in value]
        # numbers
        elif re.match(r'^-?\d+$', value):
            value = int(value)
        elif re.match(r'^-?\d+$', value):
            value = int(value)
        elif re.match(r'^-?(\d+\.\d*|\d*\.\d+)$', value):
            value = float(value)
        elif re.match(r'^-?(\d+\.\d*|\d*\.\d+|\d+)e-?\d+$', value):
            try:
                value = float(value)
            except ValueError:
                logging.warn(f"Tried to convert {value} to float. Validate regex.")
                pass
        # booleans
        elif re.match(r'^([tT]rue)$', value):
            value = True
        elif re.match(r'^([fF]alse)$', value):
            value = False
        # strings
        elif re.match(r'^\"(.+)\"$', value):
            value = str(value[1:-1])
        elif re.match(r'\s*None\s*', value):
            value = None
        elif re.match(r'\s', value):
            value = str(value)
        trainer_args[key] = value
    return trainer_args


class ArgParser:
    argparser = argparse.ArgumentParser()
    configparser = configparse.ConfigParser(interpolation=PathInterpolation())

    def __init__(self) -> None:
        self.argparser.add_argument(
            '--config', '-c', type=str, required=True,
            help='Path to configuration file'
        )
        self.argparser.add_argument(
            '--cpu', action='store_true',
            help='Whether to use the CPU or not'
        )
        self.argparser.add_argument(
            '--mps', action='store_true',
            help='Whether to use the M1 Mac MPS or not'
        )
        self.argparser.add_argument(
            '--new_data', action='store_true', default=False,
            help='Whether to generate new datasets or not'
        )
        self.argparser.add_argument(
            '--deepspeed', type=str, required=False, default=None,
            help='Whether to use deepspeed or not'
        )
        self.argparser.add_argument(
            '--fsdp', type=str, required=False, default=None,
            help='Whether to use fsdp or not'
        )
        args, _ = self.argparser.parse_known_args()
        self.args = args
        self.path = args.config

    def parse_config_file(self) -> Tuple[TrainingArguments, Dict[str, str]]:
        if os.path.dirname(self.path) == '':
            self.path = os.path.join('configs', self.path)
        with open(self.path, 'r') as config_file:
            self.configparser.read_file(config_file)
        sections = self.configparser.sections()
        # check for errors
        if 'Trainer Arguments' not in sections:
            raise ConfigSyntaxError(".conf file must contain section called Trainer Arguments",
                {'filename': self.path}
            )
        if 'output_dir' not in self.configparser['Trainer Arguments']:
            raise ConfigSyntaxError(".conf file must contain the argument output_dir",
                {'filename': self.path}
            )
        if 'batch_size' in self.configparser['Trainer Arguments']:
            if 'eval_batch_size' in self.configparser['Trainer Arguments'] or 'train_batch_size' in self.configparser['Trainer Arguments']:
                logging.warn("Provided too many batch_size arguments. Defaulting to argument in 'batch_size'")
            self.configparser['Trainer Arguments']['train_batch_size'] = self.configparser['Trainer Arguments']['batch_size']
            self.configparser['Trainer Arguments']['eval_batch_size'] = self.configparser['Trainer Arguments']['batch_size']
        if self.configparser['Trainer Arguments']['eval_batch_size'] != self.configparser['Trainer Arguments']['train_batch_size']:
            raise ConfigSyntaxError("eval_batch_size and train_batch_size must be the same. Consider using batch_size instead",
                {'filename': self.path}
            )
        # more linting to ensure intended defaults
        if 'do_train' not in self.configparser['Trainer Arguments']:
            logger.warning(
                re.sub(r'\s*\n\s*', ' ',
                """do_train not found in config file. I'm assuming you wanted to train
                   a model, so 'do_train' has been set to True. If you did not want to 
                   train a model, please set do_train to False in the config file.
                """)
            )
            self.configparser['Trainer Arguments']['do_train'] = True
        if 'do_eval' not in self.configparser['Trainer Arguments']:
            logger.warning(
                re.sub(r'\s*\n\s*', ' ',
                """do_eval not found in config file. I'm assuming you wanted to train
                   a model, so 'do_eval' has been set to True. If you did not want to 
                   train a model, please set do_eval to False in the config file.
                """)
            )
            self.configparser['Trainer Arguments']['do_eval'] = True

        trainer_args = parse_dict_types(self.configparser['Trainer Arguments'])
        
        # small fixes to save space in config file
        for x in ['eval_batch_size', 'train_batch_size']:
            trainer_args['per_device_' + x] = trainer_args.pop(x)

        trial_config = parse_dict_types(self.configparser['Training Configuration'])

        if self.args.new_data:
            trial_config['generate_new_data'] = True

        if 'Model Configuration' in sections:
            model_config = parse_dict_types(self.configparser['Model Configuration'])
        else:
            model_config = {}

        # correct for fp16 issues
        if 'fp16' in trainer_args and trainer_args['fp16']:
            if not(self.args.cpu or self.args.mps):
                model_config['fp16'] = 'half'
            else:
                logger.warning("Specified 'fp16', but training on cpu... Ignoring fp16.")
                model_config['fp16'] = 'full'
                trainer_args['fp16'] = False
        else:
            model_config['fp16'] = 'full'
        
        # correct for ctc_loss_reduction issues
        if 'ctc_loss_reduction' in model_config and model_config['ctc_loss_reduction'] == 'sum':
            logger.error("ctc_loss_reduction should be set to mean to ensure longer samples " +
                "don't carry too much weight...")
        else:
            logger.error("Setting default ctc_loss_reduction to mean " + 
                "(sum favors longer sequences)...")
            model_config['ctc_loss_reduction'] = 'mean'

        # parse deepspeed
        if self.args.deepspeed is not None:
            with open(self.args.deepspeed, 'r') as ds_config_file:
                ds_config = json.load(ds_config_file)
            optimizer_params = ds_config['optimizer']
            if 'optim' in trainer_args:
                logger.warning(f"Overriding optim in config file with deepspeed optimizer" +
                f"{optimizer_params['type']}")
                trainer_args.pop('optim')
                trainer_args['adam_beta1'] = optimizer_params['params']['betas'][0]
                trainer_args['adam_beta2'] = optimizer_params['params']['betas'][1]
                trainer_args['adam_epsilon'] = optimizer_params['params']['eps']
                trainer_args['weight_decay'] = optimizer_params['params']['weight_decay']
            if optimizer_params['params']['lr'] != trainer_args['learning_rate']:
                logger.warning(f"Overriding learning rate in config file with deepspeed lr" +
                f"{optimizer_params['params']['lr']}")
                trainer_args['learning_rate'] = optimizer_params['params']['lr']
            trainer_args['warmup_steps'] = ds_config['scheduler']['params']['warmup_num_steps']
            trainer_args['deepspeed'] = self.args.deepspeed
            trainer_args['per_device_train_batch_size'] = 8
            trainer_args['per_device_eval_batch_size'] = 8
            trainer_args['gradient_accumulation_steps'] = 2

        # parse mps
        trainer_args['use_mps_device'] = True if torch.backends.mps.is_available() and self.args.mps else False

        if self.args.fsdp is not None:
            trainer_args['fsdp'] = self.args.fsdp
            self.args.local_rank = int(os.environ["LOCAL_RANK"])

        args = Arguments(
            TrainingArguments(**trainer_args), 
            trial_config, 
            model_config, 
            self.args
        )

        return args


@dataclass
class Arguments:
    trainer_args: TrainingArguments
    trial_config: Dict[str, Any]
    model_config: Dict[str, Any]
    cmdline_args: argparse.Namespace

    def __contains__(self, val):
        if hasattr(self.trainer_args, val):
            return True
        elif val in self.trial_config:
            return True
        else:
            return False

    def __post_init__(self):
        train_args = set(self.trainer_args.to_dict().keys())
        trial_config = set(self.trial_config.keys())
        if len(train_args.intersection(trial_config)) > 0:
            logger.warning(
                re.sub(r'\s*\n\s*', ' ',
                f"""Found 1+ keys in trainer_args and trial_config that are the same!
                   Make sure that you retreive items from the args dataclass explicitly
                   with Arguments.trainer_args or Arguments.trial_config: {train_args.intersection(trial_config)}
                """)
            )

    def as_dict(self):
        return {
            'trainer_args': self.trainer_args,
            'trial_config': self.trial_config,
            'model_config': self.model_config,
            'cmdline_args': self.cmdline_args
        }

    def __str__(self):
        return str(self.as_dict())

    def __getitem__(self, key: str) -> Any:
        '''Shorthand get item from either model args or trainer args'''
        if key in self.trial_config:
            if key in self.trainer_args.to_dict():
                raise KeyError(
                    f"""Found two entries for {key}! Make sure {key} is not under
                        both Model Configuration and Trainer Arguments.
                     """
                )
            else:
                return self.trial_config[key]
        else:
            return getattr(self.trainer_args, key)

    def get_log_levels(self, log_level_dict: Optional[Dict[str, int]] = None) -> int: # int for log level enum
        from logging_utils import LOGGER_VARS
        level_string = self.trainer_args.log_level
        if level_string == 'passive' or level_string.upper() in LOGGER_VARS['DEFAULT_LOG_LEVELS']:
            return self.trainer_args.get_process_log_level()
        elif log_level_dict:
            if level_string.upper() in log_level_dict:
                level = log_level_dict[level_string.upper()]
                hf_level = str(round(level+1,-1)) if level == 25 else str(round(level-1,-1))
                self.trainer_args.log_level = LOGGER_VARS['DEFAULT_LOG_NAMES'][hf_level].lower()
                return level
            else:
                raise ValueError(f"Unrecognized log_level: {level_string.lower()}")
        else:
            logger.warning(
                f"log_level_dict is a required argument for {level_string.lower()}"
                "... defaulting to level_string == 'passive'"
            )
            self.trainer_args.log_level = "passive"
            return self.trainer_args.get_process_log_level()


def parse_arguments() -> Arguments:
    parser = ArgParser()
    args = parser.parse_config_file()
    return args


if __name__ == '__main__':
    pass
