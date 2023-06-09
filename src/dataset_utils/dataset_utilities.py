'''
This file contains utilities for loading and splitting datasets
'''

from datasets import (
    Dataset, 
    DatasetDict,
    ClassLabel
)
from .dataset_builder import (
    split_ds_helper,
    NewDatasetDict,
    SEED
)
from math import floor
import os
from typing import *
from transformers import Wav2Vec2Processor


''' ############ Logging ############ '''
import logging
logger = logging.getLogger("wav2vec2_logger")
''' ################################# '''


Example =  Dict[Any, Any]


def add_prefix(ex: Union[dict, list], data_dir: str):
    ex['audio'] = os.path.join(data_dir, ex['file_name'])
    del ex['file_name']
    return ex


def fix_prefix(ds: Dataset, data_dir: str) -> Dataset: 
    return ds.map(function=add_prefix, fn_kwargs=dict(data_dir=data_dir))


def make_ds(
        dataset_dict: Dict[str, List[str]],
        prefix: str,
        data_dir: str,
        path: str,
        citation: Optional[str] = None
    ) -> None:
    '''Make and save a dataset without the train/test/split'''
    def add_prefix(ex): 
        ex['speaker'] = prefix + ex['speaker']
        return ex
    dataset = Dataset.from_dict(dataset_dict)
    dataset = dataset.map(add_prefix)
    dataset = fix_prefix(dataset, data_dir)
    if citation:
        dataset.info.citation = citation
    dataset.save_to_disk(path)
    dataset.cleanup_cache_files()
    return None


def make_split(
        dataset_dict: Dict[str, List[str]],
        prefix: str,
        data_dir: str,
        path: str
    ) -> None:
    '''Make and save a dataset with the train/test/split'''
    def add_prefix(ex): 
        ex['speaker'] = prefix + ex['speaker']
        return ex
    dataset = Dataset.from_dict(dataset_dict)
    dataset = dataset.map(add_prefix)
    dataset = fix_prefix(dataset, data_dir)
    speaker_labels = ClassLabel(names=list(set(dataset['speaker'])))
    dataset = dataset.cast_column('speaker', speaker_labels)
    print(f"SAVING TO: {path}...\n")
    ds = split_ds_helper(dataset, 'speaker')
    ds.save_to_disk(path)
    return None


def test_dataset(ds: Dataset):
    '''Test to see the dataset is working'''
    try:
        if isinstance(ds, DatasetDict):
            x = ds['train'][0]
        else:
            x = ds[0]
    except OSError:
        print("Can't find the file for ds['train'][0]!")


def stratified_slice(
        ds: Union[NewDatasetDict, DatasetDict], 
        pct: float = 10, 
        seed: int = SEED
    ) -> Union[NewDatasetDict, DatasetDict]:
    '''Obtain a random stratified sample of a dataset (for debugging)'''
    if isinstance(ds, DatasetDict):
        new_dict = {}
        for split in ds:
            pct = pct/100 if pct > 1.0 else pct
            new_dict[split] = ds[split].train_test_split(test_size=pct, stratify_by_column='speaker', 
                                                         shuffle=True, seed=seed)['test']
        return NewDatasetDict(new_dict, ds.info, ds.cache_dir)
    else:
        pct = pct/100 if pct > 1.0 else pct
        ds.train_test_split(pct, shuffle=True, stratify_by_column='speaker', seed=seed)['test']
        return ds


if __name__ == '__main__':
    pass