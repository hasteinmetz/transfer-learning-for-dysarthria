#!/bin/env python

'''
This script creates a metadata file for the TORGO database
and then generates a train/val/test split

UASpeech files:
    Torgo
    |
    --- speaker
        |
        --- session
            (Usually 1-3 sessions)
            |
            --- prompts 
            |
            --- wav (Head and Array)
    |
    --- README
        |
        --- (Copied rom TORGO website)
'''

from dataset_utils import (
    make_ds
)
import os
import logging
import argparse
import re
from typing import *
from tqdm import tqdm


def prompt_filter(text: str, tfile: str):
    if (re.search(r'\[.*\]', text) is not None or 
        re.search(r'input/.*\.je?pg', text) is not None
        ):
        logging.warning(f"Excluding {tfile} since it is a prompt ({text})")
        return None
    elif re.search(r'xxx', text):
        logging.warning(f"Excluding {tfile} since it just {text}")
        return None
    else:
        return re.sub(r'[^\w\s]', '', text).lower()


def walk_through_trimmed_corpus(corpus_path: str) -> None:
    '''Walk through the trimmed corpus which is structured different'''
    speaker, wavs, transcripts, gender = [], [], [], []
    speaker_dirs = os.listdir(os.path.join(corpus_path, 'trimmed'))
    for s in tqdm(speaker_dirs):
        if not re.match(r'[MF]C?0[1-5]', s):
            continue
        else:
            speaker_dir = os.path.join(corpus_path, 'trimmed', s)
            for audio_file in os.listdir(speaker_dir):
                if audio_file == ".DS_Store":
                    continue
                parsed_file = audio_file.split("_")
                orig_wav_path = parsed_file[-1]
                prompt_file = orig_wav_path.replace(".wav", ".txt")
                if os.path.exists(os.path.join(corpus_path, s, f"Session{parsed_file[2][-1]}")):
                    session_dir = os.path.join(corpus_path, s, f"Session{parsed_file[2][-1]}")
                else:
                    session_no = int(parsed_file[2][-1]) - 1
                    session_dir = os.path.join(corpus_path, s, 
                                               f"Session{session_no}_{parsed_file[2][-1]}")
                    if not os.path.exists(session_dir):
                        logging.warning(f"No dir {session_dir}")
                        continue
                t_file = os.path.join(session_dir, 'prompts', prompt_file)
                if os.path.exists(t_file):
                    with open(t_file, 'r') as f:
                        text = f.read().strip()
                        processed_text = prompt_filter(text, t_file)
                        if processed_text is None:
                            continue
                        else:
                            transcripts.append(processed_text)
                            speaker.append(s)
                            wav_file = os.path.join(corpus_path, 'trimmed', s, audio_file)
                            wavs.append(wav_file)
                            gender.append(s[0])
                else:
                    logging.warning(f"Couldn't find text file {t_file}")
    native_lang = ["EN"] * len(gender)
    return dict(file_name=wavs, transcript=transcripts, speaker=speaker, 
                gender=gender, l1=native_lang)


def walk_through_corpus(corpus_path: str, noise_reduce: bool = True) -> None:
    '''Walk through the TORGO corpus repo and collect 
    the wav files and prompt files'''
    speaker, wavs, transcripts, gender = [], [], [], []
    speaker_dirs = os.listdir(corpus_path)
    for s in speaker_dirs:
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
                    if os.path.isdir(os.path.join(session_dir, 'wav_arrayMic')):
                        wav_path = os.path.join(session_dir, 'wav_arrayMic')
                    else:
                        wav_path = os.path.join(session_dir, 'wav_headMic')
                        logging.warning(f"No arrayMic: Can't find arrayMic directory in {session_dir}")
                    if noise_reduce:
                        wav_path = os.path.join(corpus_path, 'noisereduce', s, 
                                                session, os.path.basename(wav_path))
                    prompt_path = os.path.join(session_dir, 'prompts')
                    for fname in os.listdir(wav_path):
                        m = re.match(r'^(\d{4}.*)\.wav', fname)
                        if m:
                            t_file = os.path.join(prompt_path, m.group(1)) + ".txt"
                            if os.path.exists(t_file):
                                with open(t_file, 'r') as f:
                                    text = f.read().strip()
                                processed_text = prompt_filter(text, t_file)
                                if processed_text is None:
                                    continue
                                else:
                                    transcripts.append(processed_text)
                                    speaker.append(s)
                                    wav_file = os.path.join(wav_path, fname)
                                    wavs.append(wav_file)
                                    gender.append(s[0])
                            else:
                                # print(f"Couldn't find text file {t_file}")
                                logging.warning(f"Couldn't find text file {t_file}")
                        else:
                            print(f"Couldn't find match for speaker "
                                  f"{os.path.basename(speaker_dir)}, " 
                                  f"session {os.path.basename(session_dir)}, " 
                                  f"{os.path.basename(fname)}")
                            logging.warning(f"Couldn't find match for speaker "
                                            f"{os.path.basename(speaker_dir)}, " 
                                            f"session {os.path.basename(session_dir)}, " 
                                            f"{os.path.basename(fname)}")
    native_lang = ["EN"] * len(gender)
    return dict(
        file_name=wavs,
        transcript=transcripts,
        speaker=speaker,
        gender=gender,
        l1=native_lang
    )
            

def read_args():
    '''Read the input arguments for the split'''
    args = argparse.ArgumentParser()
    args.add_argument(
        '--corpus_path', 
        help='The full path to the corpus with the data to process', 
        type = str
    )
    return args


def generate_torgo_split(corpus_path, noise_reduce: bool = True, trimmed: bool = False):
    # set up logging
    log_file = os.path.join(corpus_path, 'create_split.log')
    logging.basicConfig(
        filename=log_file, 
        encoding='utf-8', 
        level=logging.DEBUG
    )
    if trimmed:
        dataset_dict = walk_through_trimmed_corpus(corpus_path)
        path = os.path.join(os.path.dirname(corpus_path), 'processed_data_trimmed', 'torgo_whole')
    else:
        dataset_dict = walk_through_corpus(corpus_path, noise_reduce)
        path = os.path.join(os.path.dirname(corpus_path), 'processed_data', 'torgo_whole')
    citation="""@article{10.1007/s10579-011-9145-0,
    author = {Rudzicz, Frank and Namasivayam, Aravind Kumar and Wolff, Talya},
    title = {The TORGO Database of Acoustic and Articulatory Speech from Speakers with Dysarthria},
    year = {2012},
    issue_date = {December 2012},
    publisher = {Springer-Verlag},
    address = {Berlin, Heidelberg},
    volume = {46},
    number = {4},
    issn = {1574-020X},
    url = {https://doi.org/10.1007/s10579-011-9145-0},
    doi = {10.1007/s10579-011-9145-0},
    abstract = {This paper describes the acquisition of a new database of dysarthric speech in terms of aligned acoustics and articulatory data. This database currently includes data from seven individuals with speech impediments caused by cerebral palsy or amyotrophic lateral sclerosis and age- and gender-matched control subjects. Each of the individuals with speech impediments are given standardized assessments of speech-motor function by a speech-language pathologist. Acoustic data is obtained by one head-mounted and one directional microphone. Articulatory data is obtained by electromagnetic articulography, which allows the measurement of the tongue and other articulators during speech, and by 3D reconstruction from binocular video sequences. The stimuli are obtained from a variety of sources including the TIMIT database, lists of identified phonetic contrasts, and assessments of speech intelligibility. This paper also includes some analysis as to how dysarthric speech differs from non-dysarthric speech according to features such as length of phonemes, and pronunciation errors.},
    journal = {Lang. Resour. Eval.},
    month = {dec},
    pages = {523–541},
    numpages = {19},
    keywords = {Dysarthria, Articulation, Speech}
    """  + "}"
    make_ds(dataset_dict, "T", corpus_path, path, citation=citation)


if __name__ == '__main__':
    args = read_args()
    args = args.parse_args()
    if len(vars(args)) != 1:
        print("Invalid number of arguments provided. Usage details:")
        args.print_help()
    else:
        generate_torgo_split(args.corpus_path)
