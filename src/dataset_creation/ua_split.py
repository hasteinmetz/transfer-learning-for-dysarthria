#!/bin/env python

'''
This script splits the data into train/test/val splits.
It saves the data for later use.
'''

from datasets import (
    ClassLabel
)
import datasets
import argparse
import os
from dataset_utils import (
    fix_prefix,
    split_ds_helper
)
from typing import Optional


def split_uaspeech(csv_file: str, data_dir: str, path: str):
    '''Split a dataset into train/val/test and save the results'''
    dataset = datasets.load_dataset("csv", data_files=csv_file)['train']
    def add_prefix(ex): 
        ex['speaker'] = "U" + ex['speaker']
        return ex
    dataset = dataset.map(add_prefix, keep_in_memory=True)
    speaker_labels = ClassLabel(names=list(set(dataset['speaker'])))
    dataset = dataset.cast_column('speaker', speaker_labels)
    dataset = fix_prefix(dataset, data_dir)
    ds = split_ds_helper(dataset, 'speaker')
    ds.save_to_disk(path)
    return None


def make_uaspeech(csv_file: str, data_dir: str, path: str, citation: Optional[str] = None) -> None:
    '''Make the dataset'''
    def add_prefix(ex): 
        ex['speaker'] = "U" + ex['speaker']
        return ex
    dataset = datasets.load_dataset("csv", data_files=csv_file)['train']
    dataset = dataset.map(add_prefix, keep_in_memory=True)
    dataset = fix_prefix(dataset, data_dir)
    if citation:
        dataset.info.citation = citation
    dataset.save_to_disk(path)
    dataset.cleanup_cache_files()
    return None


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


def generate_uaspeech_split(corpus_path, csv_file, trimmed: bool = False):
    if not os.path.isabs(csv_file):
        csv_file = os.path.join(
            corpus_path, 
            os.path.basename(csv_file)
        )
    if trimmed:
        path = os.path.join(os.path.dirname(corpus_path), 'processed_data_trimmed', 'uaspeech_whole')
    else:
        path = os.path.join(os.path.dirname(corpus_path), 'processed_data', 'uaspeech_whole')
    citation = """@article{098866f233a24ff19c0c06647f23b336,
    title = "Dysarthric speech database for universal access research",
    abstract = "This paper describes a database of dysarthric speech produced by 19 speakers with cerebral palsy. Speech materials consist of 765 isolated words per speaker: 300 distinct uncommon words and 3 repetitions of digits, computer commands, radio alphabet and common words. Data is recorded through an 8-microphone array and one digital video camera. Our database provides a fundamental resource for automatic speech recognition development for people with neuromotor disability. Research on articulation errors in dysarthria will benefit clinical treatments and contribute to our knowledge of neuromotor mechanisms in speech production. Data files are available via secure ftp upon request.",
    keywords = "Cerebral palsy, Dysarthria, Speech recognition",
    author = "Heejin Kim and Mark Hasegawa-Johnson and Adrienne Perlman and Jon Gunderson and Thomas Huang and Watkin, {Kenneth Lloyd} and Simone Frame",
    year = "2008",
    language = "English (US)",
    pages = "1741--1744",
    journal = "Proceedings of the Annual Conference of the International Speech Communication Association, INTERSPEECH",
    issn = "2308-457X",
    note = "INTERSPEECH 2008 - 9th Annual Conference of the International Speech Communication Association ; Conference date: 22-09-2008 Through 26-09-2008",  
    """ + "}"
    make_uaspeech(csv_file, corpus_path, path, citation=citation)


if __name__ == '__main__':
    args = read_args()
    args = args.parse_args()
    if len(vars(args)) != 2:
        print("Invalid number of arguments provided. Usage details:")
        args.print_help()
    else:
        generate_uaspeech_split(args.corpus_path, args.csv_file)
