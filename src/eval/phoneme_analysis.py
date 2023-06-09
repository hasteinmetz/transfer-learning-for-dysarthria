#!/bin/env python

'''
This file contains functions to discover the phonotactic contraints (and coarticulation patterns)
observed in the model outputs
'''

import os, re, json
import pandas as pd
from evaluate import load
from phonemizer import phonemize
from panphon.distance import Distance
from dataclasses import dataclass
from tqdm import tqdm
import asyncio
from typing import *

DataFrameGroupBy = pd.core.groupby.generic.DataFrameGroupBy
file_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(file_dir, 'intelligibility_data.json'), 'r') as intlfile:
    INTL = json.load(intlfile)
    UAINTL = INTL['UASpeech']
    TOINTL = INTL['TORGO']

phonemefile = os.path.join(os.path.dirname(__file__), 'phonemes.json')
with open(phonemefile, 'r') as pfile:
    PHONEMES = json.load(pfile)
CV = {}
for c in PHONEMES['consonants']:
    CV[c] = 'C'
for v in PHONEMES['vowels']:
    CV[v] = 'V'
CV["ɚ"] = "VC"

CER = load('cer')


def cvify(expression: str):
    result = ""
    for character in expression:
        if character in CV:
            result += CV[character]
        else:
            result += character
    result = re.sub(r"([CV])ː", r"\1", result)
    return result

@dataclass
class PhonemeErrorRate:
    F: Callable = Distance().weighted_feature_edit_distance

    async def _compute(self, predictions: List[str], references: List[str]) -> float:
        async def weighted_per(pred, ref):
            pred = re.sub(r"\s+", "", pred)
            ref = re.sub(r"\s+", "", ref)
            phoneme_error_rate = self.F(pred, ref)
            return phoneme_error_rate
        zipped = zip(predictions, references)
        pers = await asyncio.gather(*[weighted_per(pred, ref) for pred, ref in zipped])
        reflength = len(re.sub(r"\s+", "", "".join(references)))
        return sum(pers)/reflength
    
    def compute(self, predictions: List[str], references: List[str]) -> float:
        per = asyncio.run(self._compute(predictions=predictions, references=references))
        return per
    
PER = PhonemeErrorRate()


def get_condition(ex: str) -> Literal["L2", "CTL", "DYS"]:
    '''Helper function to map speaker IDs to conditions'''
    if ex.startswith("L"):
        return "L2"
    elif re.match(r"[TU]([MF]C|C[FM])", ex) is not None:
        return "CTL"
    else:
        return "DYS"
    
def get_dataset(ex: str) -> Literal["UAS", "TOR", "L2A"]:
    '''Helper function to map speaker IDs to datasets'''
    match ex[0]:
        case "L":
            return "L2Artic"
        case "T":
            return "TORGO"
        case _:
            return "UASpeech"
        
def group_calculate(group: DataFrameGroupBy):
    '''Helper function to calculate PER given grouped object'''
    return PER.compute(predictions=group['prediction_ipa'].fillna("").to_list(), 
                       references=group['transcript_ipa'].to_list())


def detailed_per(df: pd.DataFrame, save_directory: str):
    '''Obtain detailed (by group) PER values'''
    # dataset PER
    df['condition'] = df['speaker'].map(get_condition)
    df['dataset'] = df['speaker'].map(get_dataset)
    agg_per = {}
    for condition in df['condition'].unique():
        tmpdf = df[df['condition'] == condition]
        agg_per[condition] = PER.compute(references=tmpdf['transcript_ipa'].to_list(), 
                                         predictions=tmpdf['prediction_ipa'].to_list())
    with open(os.path.join(save_directory, 'dataset_per.json'), 'w') as dataset_per:
        json.dump(agg_per, dataset_per, indent=4)
    # intelligibility PER
    agg_intl = {}
    for ds in ["UASpeech", "TORGO"]:
        tmpdf = df[(df['condition'] == "DYS") & (df['dataset'] == ds)].copy()
        tmpdf['intl'] = tmpdf['speaker'].map(lambda ex: INTL[ds][ex], na_action='ignore')
        agg_intl[condition] = tmpdf.groupby('intl').apply(group_calculate).to_dict()
        with open(os.path.join(save_directory, f'per_{ds.lower()}_severity.json'), 'w') as intl_per:
            json.dump(agg_intl, intl_per, indent=4)
    return None


def process_file(result_file: str):
    results = pd.read_csv(result_file)
    # if not all([c in results.columns for c in ['transcript_ipa', 'prediction_ipa']]):
    results['transcript_ipa'] = phonemize(results['transcript'].str.lower().to_list(), 
                                          preserve_empty_lines=True)
    results['transcript_CV'] = results['transcript_ipa'].map(cvify)
    results['prediction_ipa'] = phonemize(results['prediction'].fillna("").str.lower().to_list(), 
                                          preserve_empty_lines=True)
    results['prediction_CV'] = results['prediction_ipa'].map(cvify)
    # phoneme_error_rate = PER.compute(references=results['transcript_ipa'].to_list(), 
    #                                  predictions=results['prediction_ipa'].to_list())
    # print(f"PER ({result_file}):\n\t{phoneme_error_rate}")
    results.to_csv(result_file[:-4] + "_phonemes.csv", encoding='utf8')
    # detailed_per(results, os.path.dirname(result_file))
    by_syllable(results, os.path.dirname(result_file))
    return results


def by_syllable(df, result_path):
    '''Analyze single syllables of the UA-Speech dataset'''
    print(f"\n{df['dataset'].unique()}")
    tmp_df = df[df['dataset'].str.lower() == 'uaspeech'].reset_index(drop=True)
    tmp_df['intl'] = df['speaker'].map(
        lambda ex: UAINTL[ex] if ex in UAINTL else TOINTL[ex] if ex in TOINTL else'ctl'
    )
    s = tmp_df[tmp_df.transcript_CV.str.contains(r'^(C*V+C*)[^CV]*$')]
    s['correct'] = s['transcript_CV'] == s['prediction_CV']
    s['complex_onset'] = s['transcript_CV'].str.startswith('CC')
    # s['diphthong'] = s['transcript_CV'].str.contains(r'VV+')
    s['complex_coda'] = s['transcript_CV'].str.contains('VCC+')
    s_by_type = (s.groupby(['intl', 'complex_onset', 'complex_coda'])  # 'diphthong'])
                  .apply(lambda x: sum(x['correct'])/len(x['correct'])))
    s_by_type_count = (s.groupby(['intl', 'complex_onset', 'complex_coda'])  # 'diphthong'])
                        .apply(lambda x: len(x['correct'])))
    s_by_type = pd.concat([s_by_type, s_by_type_count], axis=1)
    s_by_type.index = s_by_type.index.map(lambda x: x if 'mid' not in x else ('hmid', x[1], x[2])) # , x[3])) 
    s_by_type = s_by_type.sort_index(level=0)
    s_by_type.to_csv(os.path.join(result_path, 'syllables.csv'))
    # # get character error rates for CV errors
    cv_error = (s.groupby(['intl', 'complex_onset', 'complex_coda'])  # 'diphthong'])
                .apply(lambda x: CER.compute(references=x['transcript_CV'].to_list(), 
                                             predictions=x['prediction_CV'].fillna("").to_list())))
    print(cv_error)


def adjusted_results_files():
    results_path, dep_path, indie_path, zs_path = 'results', 'dependent', 'independent', 'zero-shot'
    dep_models = ['base-dependent', 'finetune-dys', 'finetune-dependent', 
                  'multitask-dependent']
    indie_models = ['base-independent', 'finetune-dys-independent', 'finetune-independent', 
                    'multitask-independent-high-lr']
    zs_models = ['base-zero-shot', 'finetune-dys-zero-shot', 
                 'finetune-zero-shot', 'multitask-zero-shot']
    pbar = tqdm(zip([dep_path, indie_path, zs_path], 
                    [dep_models, indie_models, zs_models]), leave=True)
    for exp_path, models in pbar:
        pbar.set_description(f"Processing {exp_path}")
        for m in tqdm(models, desc="Evaluating models", leave=False):
            for test_set in ['validation', 'test']:
                resfile = os.path.join(results_path, exp_path, m, test_set, 'results.csv')
                process_file(resfile)


if __name__ == '__main__':
    adjusted_results_files()
