#!/bin/env python

'''
This script averages the different waveforms for speech
in the UA-Speech dataset. The dataset contains multiple
.wav files obtained from separate microphones in the array
'''

import os, sys
import re
import librosa
from tqdm import tqdm
import numpy as np
import soundfile as sf
from audioread.exceptions import NoBackendError
from typing import *

Path = str


def filter_by_utterance(ls: List[str], prefix: str) -> None:
    '''Filter a list of files depending on whether the files
    have a prefix or not
        - ls: list of files (.wav files)
        - prefix: the prefix (utterance) to filter the list by
    '''
    return list(
        filter(lambda x: x[:-7] == prefix, ls)
    )


def get_utterances(ls: List[str]) -> Set[str]:
    '''Get a set of utterances (e.g. F#_B#_C#)'''
    def get_match(x: str) -> str:
        m = re.match(r'(C?[MF]\d\d_B\d_\w+)(?=_M\d\.wav)', x)
        return m.group(1)
    return list(set(
        map(get_match, ls)
    ))


def average_channels(
        files: List[str], 
        directory: Path, 
        save_path: Path, 
        mono: bool = False
    ) -> None:
    '''Read audio files and average the channels'''
    files_fp = []
    for _f in files:
        f = os.path.join(directory, _f)
        try:
            y, _ = librosa.load(path=f, sr=16_000, mono=mono)
            files_fp.append(y)
        except sf.LibsndfileError:
            print(f"LibsndfileError: Couldn't load {f}. Check to see if .wav file is corrupted.")
        except NoBackendError:
            print(f"NoBackendError: Couldn't load {f}. Check to see if .wav file is corrupted.")
    # now fix length
    m = max([f.shape[0] for f in files_fp])
    # pad to the max length
    files_fp = [librosa.util.fix_length(f, size=m, axis=0) for f in files_fp]
    assert all([f.shape == files_fp[0].shape for f in files_fp]), "Not all audio the same length"
    mean_fp = np.sum(files_fp, axis=0)/len(files_fp)
    sf.write(save_path, mean_fp, 16_000)
    return 


def go_through_directory(data_dir: Path) -> None:
    '''Go through the UA-Speech directory and find the all
    the files for the same word, average the waveforms, and 
    save to a mono .wav file
        - data_dir: path to UA-Speech directory (top-level)
    '''
    audio_paths = os.path.join(data_dir, 'data', 'audio', 'noisereduce')
    new_dir = os.path.join(data_dir, 'data', 'audio', 'averaged')
    if not os.path.isdir(new_dir):
        os.mkdir(new_dir)
    for speaker in tqdm(os.listdir(audio_paths)):
        if speaker.startswith("."):
            continue
        _speaker_data = os.path.join(audio_paths, speaker)
        if 'data' in os.listdir(_speaker_data):
            speaker_dir = os.path.join(_speaker_data, 'data') 
        else:
            speaker_dir = _speaker_data
        new_speaker_dir = os.path.join(new_dir, speaker)
        if not os.path.isdir(new_speaker_dir):
            os.mkdir(new_speaker_dir)
        new_speaker_data = os.path.join(new_speaker_dir, 'data')
        if not os.path.isdir(new_speaker_data):
            os.mkdir(new_speaker_data)
        directory_list = os.listdir(speaker_dir)
        set_of_utterances = get_utterances(directory_list)
        for utterance in set_of_utterances:
            files = filter_by_utterance(directory_list, utterance)
            save_path = os.path.join(new_speaker_data, utterance+"_agg.wav")
            average_channels(files, speaker_dir, save_path)
    return


def generate_averaged_files(data_dir):
    assert os.path.basename(data_dir) == 'uaspeech', "make sure top-level directory is uaspeech"
    go_through_directory(data_dir)


if __name__ == '__main__':
    data_dir = sys.argv[1]
    generate_averaged_files()