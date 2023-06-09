'''
This file contains functions that can be used to conduct
more detailed analyses of the eval outputs
    - Many functions use a dataframe derived from the detailed_evaluate
      trainer method (found in modeling/train_utils.py)
Results of these more detailed analyses include charts and logs
'''

from transformers import Wav2Vec2Processor
from transformers.trainer_utils import EvalPrediction
from evaluate import load
from dataclasses import dataclass
import numpy as np
import pandas as pd
import os
import json
import re
from itertools import product as itertools_product
from .plots import save_table_as_png
from typing import *

''' ############ Logging ############ '''
import logging
logger = logging.getLogger("wav2vec2_logger")
''' ################################# '''

''' ##### Get globals from intelligibility_data.json ##### '''
data_file = os.path.join(os.path.dirname(__file__), 'intelligibility_data.json')
with open(data_file, 'r') as intl_file:
    INTELLIGIBILITY_DATA = json.load(intl_file)
''' ###################################################### '''

pretty_print = lambda x: json.dumps(x, indent=4)
def save_json(file: str, d: object):
    with open(file, 'w') as jsonfile:
        json.dump(d, fp=jsonfile, indent=4)

''' ######################## WER ######################## '''


@dataclass
class compute_wer:
    processor: Wav2Vec2Processor
    eval_wer: Callable = load('wer')
    eval_cer: Callable = load('cer')

    def __post_init__(self):
        if not(isinstance(self.processor, Wav2Vec2Processor)):
            raise TypeError(f"processor is not type Wav2Vec2Processor. Got {type(self.processor)}")

    def __call__(self, pred: EvalPrediction) -> float:
        '''Computer the WER from prediction generated by Wav2VecTrainer
            Code adapted from: https://huggingface.co/docs/transformers/tasks/asr
        '''
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        pred.label_ids[pred.label_ids == -100] = self.processor.tokenizer.pad_token_id
        
        pred_str = self.processor.batch_decode(pred_ids)
        label_str = self.processor.batch_decode(pred.label_ids, group_tokens=False)

        wer = self.eval_wer.compute(
            predictions=pred_str, 
            references=label_str
        )

        cer = self.eval_cer.compute(
            predictions=pred_str, 
            references=label_str
        )

        return {'wer': wer, 'cer': cer}

    @classmethod
    def compute(self, predictions: List[str], labels: List[str]):
        wer = self.eval_wer.compute(
            predictions=predictions, 
            references=labels
        )
        cer = self.eval_cer.compute(
            predictions=predictions, 
            references=labels
        )
        return {'wer': wer, 'cer': cer}


''' ######################## Analysis ######################## '''


def _evaluate_by_group(
        df: pd.DataFrame, 
        wer: Optional[compute_wer] = compute_wer,
        dataset: Optional[str] = None, 
        control: Optional[Union[Literal['control', 'dysarthric'], None]] = None,
        grouping: Optional[str] = None
    ) -> Union[float, pd.DataFrame]:
    '''
    Function to evaluate the WER by filtering to a specific group
        - df: dataframe derived from the detailed_evaluate method
        - wer: word error rate function
        - agg: functions to calculate metrics (e.g., mean, stdev) as strings
        - dataset: the speech dataset to filter through
        - control: whether to evaluate the control group or not
    '''

    if dataset:
        df = df[df['speaker'].str.startswith(dataset[0])]
    if control:
        assert dataset is None or dataset.startswith('T') or dataset.startswith('U'), \
            "Controls only exist in UASpeech and Torgo datasets"
        if control == 'control':
            df = df[df['speaker'].str.contains(r'(?:T[MF]C|UC[FM])\d')]
        elif control == 'dysarthric':
            df = df[df['speaker'].str.contains(r'(?:T[MF]|U[FM])\d')]
        else:
            raise ValueError("argument control only takes values of 'control' and 'dysarthric'")
    df = df.reset_index()
    df['prediction'] = df['prediction'].str.lower()
    df['transcript'] = df['transcript'].str.lower()

    if grouping:
        metrics = {'wer': {}, 'cer': {}}
        groups = df[grouping].unique()
        for group in groups:
            evaldf = df[df[grouping] == group]
            predictions, labels = evaldf['prediction'].to_list(), evaldf['transcript'].to_list()
            group_metrics = wer.compute(predictions = predictions, labels = labels)
            metrics['wer'][group] = group_metrics['wer']
            metrics['cer'][group] = group_metrics['cer']
        
        return metrics
            
    else:
        metrics = wer.compute(predictions = df['prediction'].to_list(), 
                              labels = df['transcript'].to_list())

        return metrics


def evaluate_by_group(
        df: pd.DataFrame, 
        wer: Optional[Callable] = compute_wer,
        dataset: Optional[str] = None, 
        control: Optional[str] = None,
        grouping: Optional[str] = None
    ) -> Union[None, float, pd.DataFrame]:
    '''Driver for _evaluate_by_group to add error tolerance'''
    if dataset is not None and not any(df['dataset'].str.startswith(dataset[0].lower())):
        return "Not in dataset"
    else:
        return _evaluate_by_group(df, wer, dataset, control, grouping)
    # try:
    #     return _evaluate_by_group(df, wer, dataset, control, grouping)
    # except BaseException as e:
    #     logger.watch_out(
    #         f"Could not evaluate by group. Got error:\n{e}\nContinuing script."
    #     )
    #     return "ERROR"


def get_detailed_descriptives(df: pd.DataFrame, save_directory: str) -> None:
    '''
    Function to perform a battery of analyses on the data and export the data
    to relevant IOStreams or files
        - df: dataframe derived from the detailed_evaluate method
        - save_directory: path to save files to 
    '''
    # Add intelligibility data
    def assign_intl(x):
        if x['speaker'] in INTELLIGIBILITY_DATA['UASpeech']:
            return INTELLIGIBILITY_DATA['UASpeech'][x['speaker']]
        elif x['speaker'] in INTELLIGIBILITY_DATA['TORGO']:
            return INTELLIGIBILITY_DATA['TORGO'][x['speaker']]
        else:
            return 'non-dysarthric'
    df['intl'] = df.apply(lambda x: assign_intl(x), axis=1)

    # Overall WER/CER
    logger.message(f'overall: {evaluate_by_group(df)}')

    # Each dataset's WER/CER
    datasets = ['TORGO', 'UASPEECH', 'L2Arctic']
    _means = [evaluate_by_group(df, dataset=d) for d in datasets]
    wer_by_dataset = {
        d: m if isinstance(m, pd.Series) else m for d, m in zip(datasets, _means)
    }
    logger.message(f'wer_by_dataset:\n{pretty_print(wer_by_dataset)}')
    f_path = os.path.join(save_directory, "dataset_wer.json")
    save_json(f_path, wer_by_dataset)

    # get WER/CER for dysarthric and non-dysarthric speakers by dataset
    if any(df['speaker'].str.contains(r'(?:T[MF]C|UC[FM])\d')):
        ds_by_control = itertools_product(['TORGO', 'UASPEECH', None], ['control', 'dysarthric'])
        wer_dysarthria = {}
        for d, c in ds_by_control: 
            if d is not None:
                wer_dysarthria[": ".join([d, c])] = evaluate_by_group(df, dataset=d, control=c)
            else:
                wer_dysarthria[c + " (all)"] = evaluate_by_group(df, control=c)
        wer_dysarthria = {
            k: v.to_dict() if isinstance(v, pd.Series) else v for k, v in wer_dysarthria.items()
        }
        f_path = os.path.join(save_directory, "wer_dysarthria.json")
        save_json(f_path, wer_dysarthria)
        logger.message(f'wer_dysarthria:\n{pretty_print(wer_dysarthria)}')

    # Second language WER/CER
    try_l2 = evaluate_by_group(df=df, dataset="L2Arctic", grouping='l1')
    if try_l2 != "Not in dataset":
        l2 = pd.DataFrame(try_l2)
        f_path = os.path.join(save_directory, "wer_l2.csv")
        logger.message(f'Saving to WER descriptive statistics *by L2* to: {f_path}')
        l2.to_csv(f_path)
        save_table_as_png(l2, f_path[:-3] + "png")

    # TORGO severity WER/CER
    try_torgo = evaluate_by_group(df=df, dataset="TORGO", control='dysarthric', grouping='intl')
    if try_torgo != "Not in dataset":
        torgo_sev = pd.DataFrame(try_torgo)
        f_path = os.path.join(save_directory, "wer_torgo_severity.csv")
        logger.message(f'Saving to WER descriptive statistics *by Torgo severity* to: {f_path}')
        torgo_sev.to_csv(f_path)
        save_table_as_png(torgo_sev, f_path[:-3] + "png")

    # UASPEECH severity WER/CER
    try_uaspeech = evaluate_by_group(df=df, dataset="UASPEECH", control='dysarthric', 
                                     grouping='intl')
    if try_uaspeech != "Not in dataset":
        uaspeech_sev = pd.DataFrame(try_uaspeech)
        f_path = os.path.join(save_directory, "wer_uaspeech_severity.csv")
        logger.message(f'Saving to WER descriptive statistics *by UA-Speech severity* to: {f_path}')
        uaspeech_sev.to_csv(f_path)
        save_table_as_png(uaspeech_sev, f_path[:-3] + "png")


def uaspeech_unseen(df: pd.DataFrame, uaspeech_metadata: str, save_directory: str, 
                    wer: Optional[compute_wer] = compute_wer) -> None:
    '''Evaluate the test data for just unseen words in the UA-Speech dataset'''
    metadata = pd.read_csv(uaspeech_metadata)
    unseen_words = metadata[metadata.file_name.str.contains(r'UW\d+_agg\.wav',regex=True)]
    unseen_words = unseen_words['transcript'].unique().tolist()
    df_filtered = df[df['transcript'].str.lower().isin(unseen_words)]
    metrics_all = wer.compute(predictions = df_filtered['prediction'].to_list(),
                              labels = df_filtered['transcript'].to_list())
    df_dys = df_filtered[df_filtered['speaker'].str.contains(r'(?:T[MF]|U[FM])\d')]
    metrics_dys = wer.compute(predictions = df_dys['prediction'].to_list(),
                              labels = df_dys['transcript'].to_list())
    metrics = {"all": metrics_all, "dysarthria": metrics_dys}
    f_path = os.path.join(save_directory, "unseen_words.json")
    save_json(f_path, metrics)


def wer_by_results_file(results_path: str) -> Dict[str, float]:
    results = pd.read_csv(results_path)
    wer = load('wer')
    cer = load('cer')
    results = (results
               .drop(['Unnamed: 0'], axis=1)
               .fillna('')
               )
    word_error = wer.compute(predictions=results['prediction'].to_list(), 
                             references=results['transcript'].to_list())
    char_error = cer.compute(predictions=results['prediction'].to_list(), 
                             references=results['transcript'].to_list())
    return {'wer': word_error, 'cer': char_error}


def read_saved_df(save_path: str) -> pd.DataFrame:
    '''Load the saved results file in a way that's compatible 
       with the other functions in this module
    '''
    df = pd.read_csv(save_path, index_col=0, error_bad_lines=False)
    df[['transcript', 'prediction']] = df[['transcript', 'prediction']].fillna('')
    df = df.dropna(how='all')
    return df


def add_hyperparameters_to_results(model_path: str, results_path: str):
    '''Add important hyperparameters to the results file for easy reference'''
    if 'torch' not in dir():
        from torch import load as torch_load
    else:
        torch_load = lambda x: torch.load(x)
    trargs = os.path.join(model_path, 'training_args.bin')
    if os.path.exists(trargs):
        args = torch_load(trargs)
        hyper = {'lr': args.learning_rate, 'weight_decay': args.weight_decay, 
                'warmup_steps': args.warmup_steps, 'warmup_ratio': args.warmup_ratio, 
                'optim': args.optim, 'epochs': args.num_train_epochs}
    else:
        hyper = {}
    config = os.path.join(model_path, 'config.json')
    if os.path.exists(config):
        with open(config, 'r') as configfile:
            config = json.load(configfile)
        if 'loss_weight' in config:
            hyper.update({'loss_weight': config['loss_weight']})
    outfile = os.path.join(results_path, 'hyperparameters.json')
    if os.path.exists(trargs) and os.path.exists(results_path):
        save_json(outfile, hyper)
    else:
        print(f"Missing directory {results_path}")


if __name__ == '__main__':
    pass