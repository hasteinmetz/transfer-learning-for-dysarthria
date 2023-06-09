#!/bin/env python

'''This file contains functions to remove outliers when processing a dataset
    Motivation: The datasets contain audio files that contain audio files that:
        - Include audio of RAs speaking to participants
        - Include additional utterances not in annotation
        - Utterances that are too long to process with given resources
    Strategy: 
        1. Find the audio lengths (in s) and find outliers in top %5 of file
        2. Remove outliers longer than the max(top %5, a_priori_number)
            - a_priori_number is based on averages observed distribution within speaker
                => if goal is to remove as few audio files as possible
'''

if __name__ == '__main__':
    import sys
    from os.path import basename, dirname
    python_path = sys.path[0]
    if dirname(__file__) == python_path:
        sys.path.append(
            dirname(sys.path.pop(0))
        )

import librosa
import numpy as np
import pandas as pd
from dataset_utils import NewDatasetDict
from datasets import Dataset, DatasetDict
from pathlib import Path

from typing import *

''' ############ Logging ############ '''
import logging
logger = logging.getLogger("wav2vec2_logger")
''' ################################# '''


MAX_DURATION = 15


def make_audio_df(ds: Union[NewDatasetDict, DatasetDict, Dataset]) -> pd.DataFrame:
    '''Create a dataframe from a datasetdict
        - Make sure the ds is not processed!
    '''
    # get paths
    def get_duration(ex):
        ex['duration'] = librosa.get_duration(filename=ex['audio'])
        return ex
    ds = ds.map(get_duration)
    if isinstance(ds, DatasetDict):
        assert isinstance(ds['train']['audio'][0], str)
        _audio_arr = [d['audio'] for d in ds.values()] # get all values over the datasetdict
        _speakers = [d['speaker'] for d in ds.values()]
        _durations = [d['duration'] for d in ds.values()]
        audio_arr, speakers, durations = _audio_arr[0], _speakers[0], _durations[0]
        for i in range(1, len(_audio_arr)):
            audio_arr.extend(_audio_arr[i])
            speakers.extend(_speakers[i])
            durations.extend(_durations[i])
    else:
        assert isinstance(ds['audio'][0], str)
        audio_arr, speakers, durations = ds['audio'], ds['speaker'], ds['duration']
        
    return pd.DataFrame(
        {'speaker': speakers, 'path': audio_arr, 'duration': durations}
    )


def find_outliers(
        df: pd.DataFrame, 
        pct: float, 
        decoder: Optional[Callable] = None,
        stats_file: Optional[Callable] = None,
        max_duration: Optional[float] = MAX_DURATION,
    ) -> List[str]:
    '''Find the outliers and return a list of paths'''
    assert pct >= 0. and pct <= 1.
    files_to_remove = []
    stats_df = {'*speaker': [], f'*{pct}-q': [], '*mean': [], 
                '*no': [], "*std": [], "*total": []}
    
    for x in df['speaker'].unique():
        speaker_df = df[df['speaker'] == x]
        q = np.quantile(speaker_df['duration'], pct)
        m = np.mean(speaker_df['duration'])
        std_dev = speaker_df['duration'].std()
        filter_num = max(q, max_duration)      # set the maximum allowable duration to max of q or the thing allowed
        filter_df = speaker_df[speaker_df['duration'] >= filter_num]
        files_to_remove.extend(filter_df['path'].tolist())
        speaker = decoder(int(x)) if decoder else x

        stats_df['*speaker'].append(speaker)
        stats_df[f'*{pct}-q'].append(q)
        stats_df['*mean'].append(m)
        stats_df['*no'].append(filter_df['path'].shape[0])
        stats_df['*std'].append(std_dev)
        stats_df['*total'].append(speaker_df.shape[0])
    
    if decoder:
        stats_df = pd.DataFrame(stats_df)
        if stats_file:
            stats_df.to_csv(stats_file)
        else:
            logger.info(f"Duration stats:\n{stats_df.to_string()}")
    
    return files_to_remove


def filter_outliers(
        ds: NewDatasetDict, 
        pct: float,
        decoder: Optional[Callable] = None,
        stats_file: Optional[Path] = None,
        max_duration: Optional[float] = MAX_DURATION
    ) -> NewDatasetDict:
    '''Driver for find_outliers followed by ds.filter'''
    filter_files = find_outliers(make_audio_df(ds), pct, decoder, stats_file, max_duration)
    logger.info(f"Removed {len(filter_files)} files (max_duration = {max_duration})")
    print(f"Removed {len(filter_files)} files (max_duration = {max_duration})")
    return ds.filter(lambda x: not x['audio'] in filter_files)


if __name__ == '__main__':
    ds = NewDatasetDict.load_dataset_dict(
        '../data/dependent',
        '../data/processed_data',
        new=False
    )
    decoder = lambda x: ds.decode_class_label('speaker', x)
    ds = filter_outliers(ds, 0.98, decoder)
    print(ds)