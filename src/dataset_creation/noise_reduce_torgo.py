#!/bin/env python

'''
This script performs the noise_reduce algorithm on the
TORGO dataset
'''

import os, sys, re
import librosa
import soundfile as sf
import noisereduce as nr
from tqdm import tqdm
from typing import *

Path = str


def remove_noise(files: List[Path], sampling_rate: int = 16_000):
    '''
    Reduce the noise from a list of wavfiles and save the files 
    to a new noisereduce directory
    '''
    for file in tqdm(files):
        try:
            data, _ = librosa.load(path=file, sr=16_000, mono=True)
            reduced_noise = nr.reduce_noise(y=data, sr=sampling_rate)
            new_path = os.path.join(os.getcwd(), 'noisereduce', os.path.relpath(file, os.getcwd()))
            if not os.path.exists(os.path.dirname(new_path)):
                os.makedirs(os.path.dirname(new_path))
            sf.write(file=new_path, data=reduced_noise, samplerate=sampling_rate)
        except sf.LibsndfileError:
            print(f"LibsndfileError: Couldn't load {file}. Check to see if .wav file is corrupted.")
        except EOFError:
            print(f"EOFError: Couldn't load {file}. Check to see if .wav file is corrupted.")
    return
    

def go_through_directory(data_dir: Path) -> None:
    '''
    Go through the TORGO directory to collect paths to
    speech files and reduce their noise
        - data_dir: path to TORGO directory
    '''
    corpus_path = os.path.realpath(data_dir)
    os.chdir(data_dir)
    files = []
    for s in os.listdir('.'):
        if not re.match(r'[MF]C?0[1-5]', s):
            continue
        else:
            speaker_dir = os.path.join(corpus_path, s)
            sessions = os.listdir(speaker_dir)
            for session in sessions:
                if not re.match(r'Session\d', session):
                    continue
                else:
                    session_dir = os.path.join(corpus_path, s, session)
                    for subdirectory in ['wav_headMic', 'wav_arrayMic']:
                        wav_path = os.path.join(session_dir, subdirectory)
                        if os.path.isdir(wav_path):
                            for wavname in os.listdir(wav_path):
                                files.append(
                                    os.path.join(wav_path, wavname)
                                )
    return files


def reduce_torgo_noise(data_dir):
    assert os.path.basename(data_dir) == 'torgo', "make sure top-level directory is torgo"
    files = go_through_directory(data_dir)
    remove_noise(files)


if __name__ == '__main__':
    data_dir = sys.argv[1]
    reduce_torgo_noise(data_dir)