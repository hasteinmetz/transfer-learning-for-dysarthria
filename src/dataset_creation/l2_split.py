#!/bin/env python

'''
This script creates a metadata file for the TORGO database
and then generates a train/val/test split

UASpeech files:
    L2 Arctic
    |
    --- speaker
        |
        --- transcript 
        |
        --- wav
    |
    --- README
        |
        --- Contains table of native languages
    |
    --- speakers.csv
        |
        --- Manually created file of speakers
'''

from dataset_utils import (
    make_ds
)
import os
import csv
import re
import argparse
from typing import *


def get_speakers_info(file: str) -> None:
    '''Load the speakers.csv file, search through directories 
    and create metadata.csv'''
    speakers = {}
    with open(file, 'r') as speakersf:
        speaker_reader = csv.DictReader(speakersf)
        # change to dict where keys map to arrays of values
        for row in speaker_reader:
            speakers[row['speaker']] = {
                'gender': row['gender'],
                'lang': row['native_language']
            }
    return speakers


def walk_through_corpus(corpus_path: str, speakers: Dict[str, str]) -> None:
    '''Walk through the L2 corpus repo and collect 
    the wav files and transcript files'''
    speaker, wavs, transcripts, nat_lang, gender = [], [], [], [], []
    for s in speakers:
        wav_path = os.path.join(corpus_path, s, 'wav')
        transcript_path = os.path.join(corpus_path, s, 'transcript')
        for fname in os.listdir(wav_path):
            m = re.match(r'^(arctic.*)\.wav', fname)
            if m:
                speaker.append(s)
                wav_file = os.path.join(wav_path, fname)
                wavs.append(wav_file)
                nat_lang.append(speakers[s]['lang'])
                gender.append(speakers[s]['gender'])
                t_file = os.path.join(transcript_path, m.group(1)) + ".txt"
                if os.path.exists(t_file):
                    with open(t_file, 'r') as f:
                        text = f.read().strip()
                        transcripts.append(text)
                else:
                    print(f"WARNING: Couldn't find {t_file}")
    return dict(
        file_name=wavs,
        transcript=transcripts,
        speaker=speaker,
        gender=gender,
        l1=nat_lang
    )


def read_args():
    '''Read the input arguments for the split'''
    args = argparse.ArgumentParser()
    args.add_argument(
        '--corpus_path', 
        help='The full path to the corpus with the data to process', 
        type = str
    )
    args.add_argument(
        '--csv_file', 
        help='The full path or name of the csv file with speaker data', 
        type = str
    )
    return args


def generate_l2_split(corpus_path, csv_file, trimmed: bool = False):
    if not os.path.isabs(csv_file):
        csv_file = os.path.join(
            corpus_path, 
            os.path.basename(csv_file)
        )
    else:
        csv_file = csv_file
    if trimmed:
        path = os.path.join(os.path.dirname(corpus_path), 'processed_data_trimmed', 'l2arctic_whole')
    else:
        path = os.path.join(os.path.dirname(corpus_path), 'processed_data', 'l2arctic_whole')
    speakers = get_speakers_info(csv_file)
    dataset_dict = walk_through_corpus(corpus_path, speakers)
    citation = """@inproceedings{zhao2018l2arctic,
    author={Guanlong {Zhao} and Sinem {Sonsaat} and Alif {Silpachai} and Ivana {Lucic} and Evgeny {Chukharev-Hudilainen} and John {Levis} and Ricardo {Gutierrez-Osuna}},
    title={L2-ARCTIC: A Non-native English Speech Corpus},
    year=2018,
    booktitle={Proc. Interspeech},
    pages={2783–2787},
    doi={10.21437/Interspeech.2018-1110},
    url={http://dx.doi.org/10.21437/Interspeech.2018-1110}
    """  + "}"
    make_ds(dataset_dict, "L", corpus_path, path, citation=citation)


if __name__ == '__main__':
    args = read_args()
    args = args.parse_args()
    if len(vars(args)) != 2:
        print("Invalid number of arguments provided. Usage details:")
        args.print_help()
    else:
        generate_l2_split(args.corpus_path, args.csv_file)
