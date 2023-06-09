from .combine_channels import generate_averaged_files
from .metadata import create_csv_file
from .noise_reduce_torgo import reduce_torgo_noise
from .l2_split import generate_l2_split
from .torgo_split import generate_torgo_split
from .ua_split import generate_uaspeech_split
from .modded_speech_data import (make_baslines, DEFAULT_DYS_LIST, 
                                 DEFAULT_CTL_LIST, DEFAULT_MAX_DURATION)

import argparse, os


DIR_INFO = {
    "data_directory": "../../data",
    "torgo_directory_name": "torgo",
    "uaspeech_directory_name": "uaspeech",
    "l2arctic_directory_name": "l2arctic",
    "uaspeech_metadata": "metadata.csv",
    "l2arctic_speakers": "speakers.csv",
    "baseline_directory": {
        'dys': 'processed_data/baselines_dys/', 
        'ctl': 'processed_data/baselines_ctl/'
    },
    "baseline_subdirectories": {
        "dys": DEFAULT_DYS_LIST,
        "ctl": DEFAULT_CTL_LIST
    }
}


def parse_args():
    args = argparse.ArgumentParser()
    subparsers = args.add_subparsers(help='Create data to train a model or for baseline analyses?' +
                                     '(Options: "train" or "baselines")', dest='command')
    main_data = subparsers.add_parser('train', help='Generate data to train the model')
    main_data.add_argument(
        '--new_averages', action='store_true',
        help='Generate new averages of the channels for the UASpeech dataset'
    )
    main_data.add_argument(
        '--noise_reduce_torgo', action='store_true',
        help='Perform the noise reduce algorithm on the TORGO dataset'
    )
    main_data.add_argument(
        '--new_metadata', action='store_true',
        help='Generate a new CSV metadata file for the UASpeech dataset'
    )
    main_data.add_argument(
        '--use_trimmed', action='store_true',
        help='Whether or not to use the trimmed data for the study. Data trimmed using DNN-HMM.'
    )
    baselines = subparsers.add_parser('baselines', 
                                      help='Generate baselines to evaluate linguisit properties')
    baselines.add_argument(
        '--group', type=str, default='all', required=False,
        help='Whether to create baseline data for dysarthric group, control group, or both.'
    )
    baselines.add_argument(
        '--max_duration', type=float, default=DEFAULT_MAX_DURATION, required=False,
        help='The maximum duration of an audio file when filtering.'
    )
    args.set_defaults(command='train')
    parsed_args = args.parse_args()
    return parsed_args


def main_datasets(args):
    step, steps = 1, sum([args.new_averages, args.new_metadata, args.noise_reduce_torgo]) + 3
    directory = os.path.realpath(DIR_INFO['data_directory'])
    torgo, uaspeech, l2arctic = DIR_INFO['torgo_directory_name'], DIR_INFO['uaspeech_directory_name'], DIR_INFO['l2arctic_directory_name']
    uaspeech_metadata, l2arctic_speakers = DIR_INFO['uaspeech_metadata'], DIR_INFO['l2arctic_speakers']
    
    if args.use_trimmed:
        uaspeech_metadata = uaspeech_metadata.replace("metadata", "metadata-trimmed")
        if not args.new_metadata and not os.path.exists(uaspeech_metadata):
            steps += 1

    if args.new_averages:
        print(f"({step}/{steps}) Averaging channels of UA-Speech data...")
        step += 1
        generate_averaged_files(os.path.join(directory, uaspeech))
    if args.noise_reduce_torgo:
        print(f"({step}/{steps}) Applying noisereduce algorithm to TORGO data...")
        step += 1
        reduce_torgo_noise(os.path.join(directory, torgo))
    if args.new_metadata or not os.path.exists(uaspeech_metadata):
        print(f"({step}/{steps}) Creating {uaspeech_metadata} files for UA-Speech...")
        step += 1
        audio_type = 'trimmed' if args.use_trimmed else 'averaged'
        create_csv_file(working_dir=directory, fname=uaspeech_metadata, audio_type=audio_type)

    # generate splits
    print(f"({step}/{steps}) Creating dataset for UA-Speech")
    generate_uaspeech_split(os.path.join(directory, uaspeech), uaspeech_metadata, args.use_trimmed)
    step += 1
    print(f"({step}/{steps}) Creating dataset for TORGO")
    generate_torgo_split(os.path.join(directory, torgo), args.noise_reduce_torgo, args.use_trimmed)
    step += 1
    print(f"({step}/{steps}) Creating dataset for L2Arctic")
    generate_l2_split(os.path.join(directory, l2arctic), l2arctic_speakers, args.use_trimmed)
    step += 1


def baselines(args):
    directory = os.path.realpath(DIR_INFO['data_directory'])
    if args.group == 'all':
        baseline_dir = [DIR_INFO['baseline_directory']['dys'], 
                        DIR_INFO['baseline_directory']['ctl']]
        subdirs = [DIR_INFO['baseline_subdirectories']['dys'], 
                   DIR_INFO['baseline_subdirectories']['ctl']]
    else:
        baseline_dir = DIR_INFO['baseline_directory'][args.group]
        subdirs = DIR_INFO['baseline_subdirectories'][args.group]
    make_baslines(directory, baseline_dir, subdirs, args.max_duration)


def main():
    args = parse_args()
    if args.command == 'train':
        main_datasets(args)
    else:
        baselines(args)


if __name__ == '__main__':
    main()