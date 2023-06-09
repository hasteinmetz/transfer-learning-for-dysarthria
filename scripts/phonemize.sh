#!/bin/sh

# go to the write directory
if [[ "$(basename $(pwd))" != "repo" ]]; then
	echo "Moving to ~/thesis/repo"
	cd ~/thesis/repo
fi

python src/eval/phoneme_analysis.py results/dependent/finetune-dependent/results.csv