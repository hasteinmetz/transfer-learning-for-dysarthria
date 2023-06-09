#!/bin/env python

'''
Script to create a metadadta.csv file so that HuggingFace can 
parse the transcriptions

UASpeech files:
    Audio
    |
    --- speaker
        |
        --- Utterance (~5000)
    mlf
    |
    --- speaker
        |
        --- One .mlf file
            (In file: 
                "*/[file_name].lab"
                WORD
                .    
            )
'''

import os, sys, re
import csv


def create_csv_mlf(working_dir: str, fname: str) -> None:
    '''NOTE: Make sure to be one directory above UASpeech'''
    os.chdir(working_dir)
    speakers = [
        s for s in os.listdir('uaspeech/data/mlf') if re.match(r'C?(M|F)\d\d', s)
    ]
    rows = []
    for speaker in speakers:
        with open(f'uaspeech/data/mlf/{speaker}/{speaker}_word.mlf') as infile:
            wordsfile = infile.readlines()
        lines = iter(wordsfile)
        for line in lines:
            if re.match(r'\"\*\/.*\.lab', line):
                m = re.search(r'(?<=\"\*/)(?P<file_name>.*)(?=\.lab)', line)
                file = m.group('file_name') if m else "None"
                filename = f"data/audio/noisereduce/{speaker}/{file}."
                word = next(lines)
                word = word.strip()
                rows.append([filename, word])
    with open(fname, 'w', encoding='UTF8', newline='') as outfile:
        writer = csv.writer(outfile)
        # write headers
        writer.writerow(['file_name','transcription'])
        writer.writerows(rows)
    return None


def create_csv_file(working_dir: str, fname: str = 'metadata.csv', audio_type: str = 'averaged') -> None:
    '''NOTE: Make sure to be one directory above UASpeech'''
    os.chdir(working_dir)
    word_file_dict = {}
    with open('uaspeech/speaker_wordlist.csv', 'r') as wordlist:
        wordreader = csv.reader(wordlist, delimiter=",")
        for row in wordreader:
            word_file_dict[row[1]] = row[0]
    data_dir = os.path.join('uaspeech', 'data', 'audio', audio_type)
    rel_path = 'uaspeech/'
    global_metadata_rows = []
    # Iterate over speakers and add to both local metadata file and global metadata file
    print("Speakers:", os.listdir(data_dir))
    for speaker_d in os.listdir(data_dir):
        if speaker_d.startswith('.'):
            continue
        speaker_dir = os.path.join(data_dir, speaker_d)
        speaker_metadata_rows = []
        for (root, _, files) in os.walk(speaker_dir):
            if len(files) < 3:
                continue
            else:
                root_path = os.path.relpath(root, rel_path)
                for speechf in files:
                    global_speechf = os.path.join(root_path, speechf)
                    new_rel_path = os.path.join(speaker_dir)
                    local_speechf = os.path.relpath(os.path.join(root, speechf), new_rel_path)
                    file_split = speechf[:-4].split("_")
                    base_path, block, word, mic = file_split[0], file_split[1], file_split[2], file_split[3]
                    gender = 'M' if 'M' in str(speaker_d) else 'F'
                    speaker = os.path.basename(base_path)
                    if word[0] == 'U':
                        word = "_".join([block, word])
                    speaker_metadata_rows.append([local_speechf, word_file_dict[word], speaker, gender, "EN"])
                    global_metadata_rows.append([global_speechf, word_file_dict[word], speaker, gender, "EN"])
        speaker_metadata_file = os.path.join(speaker_dir, os.path.basename(fname))
        with open(speaker_metadata_file, 'w', encoding='UTF8', newline='') as outfile:
            writer = csv.writer(outfile)
            # write headers
            writer.writerow(['file_name','transcript', 'speaker', 'gender', 'l1'])
            writer.writerows(speaker_metadata_rows)
    with open(os.path.join('uaspeech', fname), 'w', encoding='UTF8', newline='') as outfile:
        writer = csv.writer(outfile)
        # write headers
        writer.writerow(['file_name','transcript', 'speaker', 'gender', 'l1'])
        writer.writerows(global_metadata_rows)
    return None


def create_csv():
    working_dir, fname = sys.argv[1], sys.argv[2]
    create_csv_file(working_dir, fname)


if __name__ == '__main__':
    create_csv()