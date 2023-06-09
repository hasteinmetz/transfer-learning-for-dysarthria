from typing import *
import datasets
from datasets import (
    Dataset, 
    DatasetDict,
    DatasetInfo,
    ClassLabel,
    Value,
    load_dataset
)
# from process import filter_outliers
import os
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd

''' ############ Logging ############ '''
import logging
logger = logging.getLogger("wav2vec2_logger")
''' ################################# '''


# global variables
SEED=2022
DEFAULT_DATA_DIR = os.path.join(os.path.expanduser("~"), "thesis", "data")
DEFAULT_PROCESSED_DIR = os.path.join(DEFAULT_DATA_DIR, 'processed_data')
DEMOGRAPHICS_FILE = os.path.join(DEFAULT_DATA_DIR, 'speaker_demographics.csv')
# np.random.default_rng(2022) # set the seed for numpy-based functions


if "HF_DATASETS_CACHE" in os.environ:
    DEFAULT_CACHE_DIR = os.environ["HF_DATASETS_CACHE"]
else:
    DEFAULT_CACHE_DIR = os.path.join(os.path.expanduser("~"), 
                                     "thesis", ".cache", "huggingface", "datasets")


def contains_regex(pattern: str, s: str):
    '''Helper function to search for regex in a string'''
    if re.search(pattern, s):
        return True
    else:
        return False


def determine_split_type(dataset_path: str) -> str:
    '''Helper function to determine the dataset split type from the dataset_path basename.
       Evaluated using ".startwith" function
    '''
    if dataset_path.startswith("dependent"):
        return "dependent"
    elif dataset_path.startswith("independent"):
        return "independent"
    elif dataset_path.startswith("zero-shot"):
        return "zero-shot"
    else:
        raise ValueError(f"dataset_path must start with one of ['dependent', 'independent', " 
                         f"f'zero-shot']. ours is {dataset_path}")


##### WRAPPER AROUND NewDatasetDict class to preserve parent methods
def MethodWrapper(cls):
    '''Wrapper around class to return correct subclass for certain methods
    Code adapted from Stack Exchange answer: https://stackoverflow.com/q/53134773/'''
    def fix_func(func_name: Callable):
        def func(self, *args, **kwargs):
            method = getattr(super(cls, self), func_name)
            results = method(*args, **kwargs)
            return cls(
                dictionary=results,
                info=self.info,
                cache_dir=self.cache_dir
            )

        func.__name__ = func_name
        func.__qualname__ = '{}.{}'.format(cls.__qualname__, func_name)
        func.__module__ = cls.__module__

        return func

    for func_name in ['cast_column', 'filter', 'sort', 'shuffle', 'cast', 'map',
            'rename_columns', 'rename_column', 'flatten', 'set_format', 'with_transform',
            'from_csv', 'from_json', 'from_text'
        ]:
        func = fix_func(func_name)
        setattr(cls, func_name, func)
    
    return cls


def context_manager(dataset_path: Path) -> bool:
    '''check for .context_manager file to check for newly generated data
       args:
            - dataset_path: the path to the data directory for the desired
                            dataset type (dependent, independent, ...)
    '''
    context_manager_file = os.path.join(dataset_path, ".context_manager")
    new = False
    if os.path.exists(context_manager_file):
        with open(context_manager_file, 'r') as contextf:
            data_context = contextf.read().strip()
        if 'new data' in data_context:
            logger.message("Context file indicates new or changed data. " + 
                            "Reconstructing dataset.")
            new = True
            with open(context_manager_file, 'w') as contextf:
                pass # reset context file to be blank
    else:
        data_directory = os.path.dirname(context_manager_file)
        if not os.path.exists(data_directory):
            os.makedirs(data_directory)
        with open(context_manager_file, 'w') as contextf:
            pass   # create the context manager file
        new = True # assume the data is new without context file
    return new
            

@MethodWrapper
class NewDatasetDict(DatasetDict):
    '''Subclass of DatasetDict that includes customized loading and saving functions'''

    def __init__(
            self, 
            dictionary: Union[Dict[str, Dataset], DatasetDict], 
            info: Optional[DatasetInfo] = None, 
            cache_dir: str = None) -> None:
        self.info = info
        self.cache_dir = cache_dir
        if isinstance(dictionary, DatasetDict):
            dictionary = {k: v for k, v in dictionary.items()}
        super().__init__(dictionary)


    @property
    def ds_splits(self) -> List[str]:
        return [k for k in self]


    @classmethod
    def load_dataset_dict(
            cls, 
            dataset_path: Optional[str], 
            processed_data_path: Optional[str] = DEFAULT_PROCESSED_DIR,
            new: Optional[bool] = False,
            cache_dir: Optional[str] = None
        ):
        '''Read the split TORGO, UA-Speech, and L2Arctic datasets from a specified folder'''
        split_type = determine_split_type(os.path.basename(dataset_path))
        if 'trimmed' in processed_data_path and '_trimmed' not in dataset_path:
            dataset_path += '_trimmed'

        def create_new_ds():
            directories = os.listdir(processed_data_path)
            valid_dataset_prefixes = ["torgo", "uaspeech", "l2arctic"]
            dataset_paths = {}
            for d in directories:
                if d.split("_")[0] in valid_dataset_prefixes:
                    dataset_paths[d] = os.path.join(processed_data_path, d)
            ds = dataset_split_driver(dataset_paths, split_type)
            logger.warn(f"Saving new dataset to {dataset_path}")
            ds.save_dataset(dataset_path)
            ds.cleanup_cache_files()
            return cls.read_datasets(dataset_path, cache_dir)

        if cache_dir is not None:
            logger.info(f"Default cache {DEFAULT_CACHE_DIR}")
            cache_dir = os.path.join(DEFAULT_CACHE_DIR, cache_dir)
            if not os.path.isdir(cache_dir):
                os.makedirs(cache_dir)
            logger.info(f"Overriding default cache directory with {cache_dir}")

        new = context_manager(dataset_path)

        if new is False:
            try:
                ds = cls.read_datasets(dataset_path, cache_dir)
            except FileNotFoundError:
                processed_data_path = os.path.realpath(processed_data_path)
                warning = ("Dataset not found. Creating a new one from data in "
                f"{processed_data_path}. Save it with datasetdict.save_dataset(path) "
                "for faster processing")
                logger.warn(warning)
                ds = create_new_ds()
            return ds
        else:
            processed_data_path = os.path.realpath(processed_data_path)
            logger.warn(
                f"Creating a new dataset from data in {processed_data_path}."
            )
            ds = create_new_ds()
            return ds


    @classmethod
    def read_datasets(cls, path: str, cache_dir: str = None):
        '''Read train/val/test splits from a directory'''
        path = os.path.realpath(path)
        info_file_path = os.path.join(path, "dataset_info.json")
        with open(info_file_path, 'r') as infofile:
            info = json.load(infofile)
        info = DatasetInfo(**info)
        datasets = load_dataset( # NOTE: from_json method results in weird hashing
            path='json',
            name=os.path.basename(path),
            data_dir=path, 
            features=info.features,
            cache_dir=cache_dir,
            save_infos=True
        )
        logger.info(f"Using dataset {datasets}")
        return cls(datasets, info, cache_dir)


    def save_dataset(self, path: str) -> None:
        '''Save the dataset dictionary to a particular location'''
        if not os.path.exists(path):
            os.makedirs(path)
        if self.info:
            self.info.write_to_directory(path, pretty_print=True)
        else:
            logger.warn("Oops. No dataset info. Might result in errors later on.")
        for split in self.keys():
            ds_path = os.path.join(
                os.path.realpath(path), 
                split + ".json"
            )
            ds = self.__getitem__(split)
            ds.to_json(ds_path)
        dirname = os.path.join(path)
        dictinfo_file = os.path.join(dirname, "dataset_dict.json")
        with open(dictinfo_file, 'w') as dictfile:
            json.dump({"splits": list(self.keys())}, dictfile)
        return None


    def decode_class_label(self, feature: str, x: Union[int, List[int]]) -> Union[int, List[int]]:
        '''Return a (list of) decoded class label(s)'''
        if isinstance(x, list):
            return [self.info.features[feature].int2str(i) for i in x]
        else:
            return self.info.features[feature].int2str(x)


    def filter_datasets(self, dataset_strings: Union[str, List[str]], keep_in_memory=False):
        if isinstance(dataset_strings, str):
            dataset_strings = [dataset_strings]
        def in_dataset_list(s):
            return any([s.startswith(ds.lower()[0:2]) for ds in dataset_strings])
        return self.filter(lambda ex: not(in_dataset_list(ex['dataset'])))


    def select_l2_speakers(self, speakers: List[str], keep_in_memory=False):
        '''Select/sample specific speakers from the L2Arctic dataset'''
        decode = lambda x: self.decode_class_label('speaker', x['speaker'])
        return self.filter(lambda x: decode(x) in speakers or not decode(x).startswith("L"), 
                           keep_in_memory=keep_in_memory)


    def filter_controls(self, keep_in_memory=False):
        decode = lambda x: self.decode_class_label('speaker', x['speaker'])
        return self.filter(lambda x: not(contains_regex(r'(T[MF]C|UC[FM])\d', decode(x))) or x['dataset'].startswith("l2"),
                           keep_in_memory=keep_in_memory)
    

    def get_counts(self, keep_in_memory=True):
        counts = {}
        decode = lambda x: self.decode_class_label('speaker', x['speaker'])
        counts['dys'] = self.filter(lambda x: contains_regex(r'(T[MF]|U[FM])\d', decode(x)),
                                    keep_in_memory=keep_in_memory).num_rows
        counts['l2'] = self.filter(lambda x: x['task']==1, keep_in_memory=keep_in_memory).num_rows
        counts['con'] = self.filter(lambda x: contains_regex(r'(T[MF]C|UC[FM])\d', decode(x)), 
                                    keep_in_memory=keep_in_memory).num_rows
        return counts
    

    def get_unique(self, attribute: str, decode: bool = False) -> List[Any]:
        '''Get an ordered list of unique entries in a column'''
        _unique_vals = self.unique(attribute)
        unique_vals = set()
        for ds in _unique_vals:
            unique_vals = unique_vals.union(set(_unique_vals[ds]))
        values = sorted(list(unique_vals))
        if decode:
            return [self.decode_class_label(attribute, v) for v in values]
        else:
            return values
        

    def update_features(self, feature_name: str, new_feature: Union[ClassLabel, Value]):
        '''Update a feature for all datasets in datasetdict'''
        dict_features = self.info.features[feature_name].names
        assert "".join(new_feature.names).startswith("".join(dict_features))
        self.info.features[feature_name] = new_feature
        for ds in self.keys():
            ds_feature = self[ds].info.features[feature_name].names
            assert "".join(new_feature.names).startswith("".join(ds_feature))
            self[ds].info.features.update({feature_name: new_feature})
        for ds in self.keys():
            print(ds, self[ds].info.features)
        return self


    def add_intelligibility_scores(self, intl_scores_path: str, collapse_ua_torgo: bool = True):
        '''Map intelligibility scores to the data based on scores found in the specific JSON file
           - collapse_ua_torgo: whether to map TORGO ratings on UA-Speech ones
        '''
        with open(intl_scores_path, 'r') as json_file:
            intl_scores = json.load(json_file)
        to_scores = intl_scores['TORGO_MAPPED'] if collapse_ua_torgo else intl_scores['TORGO']
        intl_scores = {**intl_scores['UASpeech'], **to_scores}
        l1_feature = self.info.features['l1']
        new_l1_feature = ClassLabel(names = [*l1_feature.names, *set(intl_scores.values())])
        self = self.update_features('l1', new_l1_feature)
        def add_intl_scores(ex):
            global checkset
            key = self.decode_class_label('speaker', ex['speaker'])
            if key in intl_scores:
                ex['l1'] = new_l1_feature.str2int(intl_scores[key])
            return ex
        self = self.map(add_intl_scores)
        return self


def read_audio_dataset(path: str) -> Dataset:
    '''Load a dataset from the dataset arrow file'''
    ds = datasets.load_from_disk(path)
    return ds


def get_classlabels(ds: Union[Dataset, DatasetDict], ls: List[str]) -> Dataset:
    '''Take a list of column names and return a dataset with those columns as classlabels'''
    for f in ls:
        classlabels = ClassLabel(names=list(set(ds[f])))
        ds = ds.cast_column(f, classlabels)
    return ds


def concatenate_info(list_of_ds: List[Dataset]) -> DatasetInfo:
    '''Take a list of datasets and obtain the dataset information'''
    return {ds.info.description: ds.info for ds in list_of_ds}


def dataset_split_driver(
        dataset_paths: Dict[str, str], 
        split_type: Literal['dependent', 'independent', 'zero-shot'],
        seed: Optional[int] = SEED
    ) -> NewDatasetDict:
    '''
    Read and combine datasets before splitting them into train/validation/test
    - split_type: whether to configure the dataset for dependent, independent or zero-shot
                  learning experiments
    '''
    data = []
    for _name, path in dataset_paths.items():
        ds = read_audio_dataset(path)
        name = _name.split("_")[0]
        dataset_name = [name] * ds.num_rows
        ds = ds.add_column(name='dataset', column = dataset_name)
        if "l2arctic" in name:
            task = [1] * ds.num_rows
            speaker_type = ['l2'] * ds.num_rows
        else:
            check_control = lambda x: True if re.search(r'([FM]C|C[MF])', x) else False
            task = [check_control(s) for s in ds['speaker']]
            task = list(map(lambda x: 2 if x else 0, task))
            speaker_type = ['control' if t == 2 else 'dysarthria' for t in task]
        ds = ds.add_column(name='task', column = task)
        ds = ds.add_column(name='speaker_type', column = speaker_type)
        ds.info.description = name
        data.append(ds)
    dataset = datasets.concatenate_datasets(data)
    # filter outliers by audio length
    # dataset = filter_outliers(dataset, 0.95, lambda x: x)
    dataset = get_classlabels(dataset, ['speaker', 'gender', 'l1']) # hardcoded for now, but can be changed
    dirname = os.path.dirname(path)
    assert all([os.path.dirname(p)==dirname for p in dataset_paths.values()])
    dataset.info.write_to_directory(dirname, pretty_print=True)
    speaker_2_str = lambda x: dataset.info.features['speaker'].int2str(x)
    if split_type == 'dependent':
        ds_dict = split_ds_helper(dataset, 'speaker', seed)
    elif split_type == 'independent':
        # load the demographic data
        np.random.default_rng(seed) # set the seed for numpy-based functions
        demographics = Dataset.from_pandas(pd.read_csv(DEMOGRAPHICS_FILE))
        category = ClassLabel(names=list(demographics.unique('category')))
        demographics = demographics.cast_column('category', category)
        # 0.25 to ensure that ALL classes are in test/dev sets
        split_speaker_ids = demographics.train_test_split(0.25, seed=seed, shuffle=True,
                                                          stratify_by_column='category') 
        # check if all categories are found in the test sets
        for cat in demographics.unique('category'):
            test_categories = [category.int2str(x) for x in split_speaker_ids.unique('category')['test']]
            assert category.int2str(cat) in test_categories, f"{cat} not in test set..." + \
                "Try raising the split percentage in dataset_builder.py: dataset_split_driver"
        train_speakers = split_speaker_ids['train'].unique('speaker')
        test_speakers = split_speaker_ids['test'].unique('speaker')
        print(f"(Independent split) Test speakers: {test_speakers}")
        train = dataset.filter(lambda x: speaker_2_str(x['speaker']) in train_speakers)
        test = dataset.filter(lambda x: speaker_2_str(x['speaker']) in test_speakers)
        testvalidation = test.train_test_split(test_size=0.5, shuffle=True, seed=seed, 
                                               stratify_by_column='speaker')
        # check if all categories are found in the test sets
        for dset in ['train', 'test']:
            dset_speakers = [speaker_2_str(x) for x in testvalidation.unique('speaker')[dset]]
            assert all([spk in dset_speakers for spk in test_speakers]), \
                f"{set(test_speakers).difference(set(dset_speakers))} missing from test set {dset}"
        ds_dict = NewDatasetDict(
            {'train': train, 'validation': testvalidation['train'], 'test': testvalidation['test']},
            info = dataset.info
        )
    elif split_type == 'zero-shot':
        train = dataset.filter(
            lambda x: contains_regex(r'(T[MF]C|UC[FM])\d', speaker_2_str(x['speaker'])) or \
                      x['dataset'].startswith("l2")
        )
        subsample = train.train_test_split(test_size=0.1, shuffle=True, 
                                           seed=seed, stratify_by_column='speaker')['test']
        _testvalidation = dataset.filter(
            lambda x: contains_regex(r'(T[MF]|U[FM])\d', speaker_2_str(x['speaker']))
        )
        testvalidation = _testvalidation.train_test_split(test_size=0.5, shuffle=True, 
                                                          seed=seed, stratify_by_column='speaker')
        testvalidation = DatasetDict({ # add subsample to the eval data (to get a sense of what's going on)
            "train": datasets.concatenate_datasets([testvalidation['train'], subsample]),
            "test": datasets.concatenate_datasets([testvalidation['test'], subsample])
        })
        ds_dict = NewDatasetDict(
            {'train': train, 'validation': testvalidation['train'], 'test': testvalidation['test']},
            info = dataset.info
        )
        print(ds_dict.num_rows)
    else:
        raise ValueError(f"Unrecognized split_type {split_type}. " + \
                         "split_type must be either 'dependent', 'independent', 'zero-shot'")
    return ds_dict


def read_dataset_splits(dataset_paths: Dict[str, str], seed=SEED) -> DatasetDict:
    '''Read and combine datasets that were already split'''
    data = []
    for name, path in dataset_paths.items():
        ds = read_audio_dataset(path)
        for ds_split in ds.values():
            ds_split.add_column('dataset', [name] * ds_split.num_rows)
        data.append(ds)
    all_datasets = {}
    for spl in ['train', 'validation', 'test']:
        all_datasets[spl] = datasets.concatenate_datasets(
            list(map(lambda x: x[spl], data))
        )
    all_datasets = DatasetDict(all_datasets)
    all_datasets = all_datasets.shuffle(seed=seed)
    return all_datasets


def load_all_dataset_splits(data_path: Path) -> DatasetDict:
    '''Read the split TORGO, UA-Speech, and L2Arctic datasets from a specified folder'''
    directories = os.listdir(data_path)
    valid_dataset_prefixes = ["torgo", "uaspeech", "l2arctic"]
    dataset_paths = {}
    for d in directories:
        if d.split("_")[0] in valid_dataset_prefixes:
            dataset_paths[d] = os.path.join(data_path, d)
    return read_dataset_splits(dataset_paths)


# NOTE: To stratify additional columns, just create a composite value
#   e.g., speaker_language

def split_ds_helper(
        ds: Dataset, 
        stratify_col: str,
        seed: Optional[int] = SEED
    ) -> NewDatasetDict:
    '''Split the dataset into train/val/test splits stratified by column values'''
    info = ds.info
    ds = ds.train_test_split(
        test_size=0.2, 
        shuffle=True, 
        seed=seed, 
        stratify_by_column=stratify_col
    )
    test = ds['test'].train_test_split(
        test_size=0.5, 
        shuffle=True, 
        seed=seed, 
        stratify_by_column=stratify_col
    )
    ds = NewDatasetDict(
        {
            'train': ds['train'],
            'validation': test['train'],
            'test': test['test']
        }, 
        info=info
    )
    return ds