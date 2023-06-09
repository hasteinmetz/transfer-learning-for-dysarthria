# script to load cmudict/L1Arctic data

import datasets
import pandas as pd
import os
import re
from typing import *
import numpy as np
from process import filter_outliers


np.random.default_rng(2022) # set the seed for numpy-based functions
DEFAULT_CTL_LIST = ['wav', 'pitch', 'rate', 'intensity_dropoff', 'intensity_mono']
DEFAULT_DYS_LIST = ['dys', 'rate_dys']
DEFAULT_GROUP = "ALL"
DEFAULT_MAX_DURATION = 10


def get_uaspeech_id(filename: str) -> str:
    '''Determine a unique ID for each prompt using the UA-Speech file naming convention'''
    return "_".join(filename.split("/")[-1].split("_")[1:3]) 


def go_through_wav_dir(wavfolders: List[str], modded_data: List[Dict[str, str]], 
                       datapath: str, prompt_data: pd.DataFrame) -> List[Dict[str, str]]:
    '''Helper function to go through a list of wav folders and extract wav file data
       Arguments:
        - wavfolders: the different modifications to the speech files to evaluate
        - modded_data: the list of data being appended to
        - datapath: the path to arrive at the current directory
        - prompt_data: dataframe containing information on prompts
    '''
    get_value_from_df = lambda pid: prompt_data.loc[pid].values[0]
    for wavfolder in wavfolders:
        wavfolder_expanded = os.path.join(datapath, wavfolder)
        for audio in os.listdir(wavfolder_expanded):
            if not re.search(r'.*\.wav$', audio):
                continue
            prompt_id = get_uaspeech_id(audio)
            if wavfolder != 'wav' and wavfolder != 'dys':
                factor_raw = re.search(r'(\d+)\.wav$', audio).group(1)
                factor = re.sub(r'(\d+)(\d?)', r'\1.\2', factor_raw)
            elif wavfolder == 'wav':
                factor = 'ctl'
            else:
                factor = 'dys'
                wavfolder = 'wav'
            audio_path = os.path.join(wavfolder_expanded, audio)
            gender = "M" if audio[1] == 'M' else 'F'
            row_entry = {'mod': wavfolder, 'factor': factor, 'task': 0, 'gender': gender,
                         'transcript': get_value_from_df(prompt_id), 
                         'id': prompt_id, 'audio': audio_path}
            modded_data.append(row_entry)
    return modded_data


def collect_data(modded_dir: str, speakers: bool, prompt_file: str, wavfolders: List[str]
                 ) -> List[Dict[str, str]]:
    '''Load into a list of dictionaries:
        - the names of the audio files 
        - corresponding prompts
        - condition & demographic information
       Arguments:
        - modded_dir (path): directory containing the modified speech
        - speakers (bool): whether or not to search through speaker files. false means that
                           the files are located in the wavfolder
        - prompt_file (path): the path to the prompt file to load
        - wavfolders: the different modifications to the speech files to evaluate
    '''
    prompts = pd.read_csv(prompt_file)
    if 'id' not in prompts.columns: # if loading uaspeech's metadata.csv 
        prompts['id'] = prompts['file_name'].map(get_uaspeech_id) 
        prompts = prompts[['id', 'transcript']].drop_duplicates()
    prompts = prompts.set_index('id')
    modded_data = []
    if speakers:
        for speaker in os.listdir(modded_dir):
            data_path = os.path.join(modded_dir, speaker)
            modded_data = go_through_wav_dir(wavfolders, modded_data, data_path, prompts)
    else:
        modded_data = go_through_wav_dir(wavfolders, modded_data, modded_dir, prompts)
    return modded_data


def create_modded_speech_dataset(modded_dir: str, speakers: bool, prompt_file: str, save_dir: str,
                                 wavfolders: List[str], max_duration: int) -> datasets.Dataset:
    '''Load dataset of the Praat-modified data:
        - the names of the audio files 
        - corresponding prompts
       Arguments:
        - modded_dir (path): directory containing the modified speech
        - speakers (bool): whether or not to search through speaker files. false means that
                           the files are located in the wavfolder
        - prompt_file (path): the path to the prompt file to load
        - wavfolders: the different modifications to the speech files to evaluate
        - save_directory (optional, path): where to save the dataset to
        - max_duration (optional, int): the max duration of the files (for filtering)
    '''
    modded_dir = os.path.realpath(modded_dir)
    data = collect_data(modded_dir, speakers, prompt_file, wavfolders)
    ds = datasets.Dataset.from_dict(pd.DataFrame(data))
    
    # map to speakers
    def speak(ex):
        ex['speaker'] = ex['audio'].split("/")[-1].split("_")[0]
        ex['basefile'] = "_".join([ex['speaker'], ex['id']])
        return ex
    ds = ds.map(speak, desc="Getting speaker names")
    
    # remove outliers to outliers file
    unmodded = ds.filter(lambda x: x['mod'] == 'wav', desc="Getting unmodded dataset")
    print(unmodded)
    unmodded_f = filter_outliers(unmodded, 0.95, max_duration=max_duration)
    removed_files = list(set(unmodded['basefile']) - set(unmodded_f['basefile']))
    print(f"Removed files: {removed_files}")
    dsf = ds.filter(lambda ex: ex['basefile'] not in removed_files, desc="Filtering outliers")
    print(f"Removed {ds.num_rows - dsf.num_rows} files in total")

    # shuffle (not super necessary)
    dsf.shuffle(seed=2022).save_to_disk(save_dir)
    
    return dsf


def make_baslines(data_dir: str, baseline_dir: Union[str, List[str]], 
                  mod_list: Union[str, List[str]], max_duration: Optional[int] = 10) -> None:
    if isinstance(baseline_dir, str):
        create_modded_speech_dataset(os.path.join(data_dir, 'uaspeech/data/audio/modified'), False, 
                                     os.path.join(data_dir, 'uaspeech/metadata.csv'), 
                                     os.path.join(data_dir, baseline_dir), mod_list, max_duration)
    else:
        assert type(baseline_dir) == type(mod_list) and len(baseline_dir) == len(mod_list), \
               (f"baseline_dir is not the type ({type(baseline_dir)}) or len ({len(baseline_dir)})",
                f" same length as mod_list ({type(mod_list)} and {len(mod_list)})")
        for baseline_dir, mod_list in zip(baseline_dir, mod_list):
            create_modded_speech_dataset(os.path.join(data_dir, 'uaspeech/data/audio/modified'), False, 
                                         os.path.join(data_dir, 'uaspeech/metadata.csv'), 
                                         os.path.join(data_dir, baseline_dir), 
                                         mod_list, max_duration)
