#!/bin/bash

# This script creates datasets for the UA-Speech, TORGO, and L2 Arctic corpora
# Your directory should be configured like the outline below. You can change the names of the 
# directories in the src/dataset_creation/__main__.py file.
#  
#   -- data
#   |   |
#   |   -- UA-Speech
#   |   |
#   |   -- TORGO
#   |   |
#   |   -- L2 Arctic
#   |
#   -- repo
#       |
#       -- src
#           |
#           -- scripts/make_all_data_splits.sh

source ~/.bashrc

# ensure that conda is activated
if [ -z "$CONDA_DEFAULT_ENV" ] || [[ $CONDA_DEFAULT_ENV != "thesis-env" ]]; then
	echo "Activating thesis-env"
	source ~/anaconda3/etc/profile.d/conda.sh
	conda activate thesis-env
	ecode=$?
else
	echo "Conda already activated"
	ecode=$?
fi

if [ $ecode -ne 0 ]; then
	echo "(check_dir code: $ecode) Issue loading conda environment. Aborting script with error status: $ecode"
	exit $ecode
else
	echo "(check_dir code: $ecode) Successfully loaded conda environment: $CONDA_DEFAULT_ENV"
fi

WORKINGDIR="$(pwd)"
if [[ "$(basename $WORKINGDIR)" != "src" ]]; then
	if [ -d "src" ]; then
		cd src
		python -m dataset_creation $@
		cd $WORKINGDIR
	else
		echo "Can't find src directory... Please execute script from top-level directory of the repository or from the src directory"
		exit 1
	fi
else
	python -m dataset_creation $@
fi

echo "Successfully created all dataset splits!"

toggle_context_managers () {
	if [ -d "../data/$1" ]; then
		if [ ! -f ./data/processed_data/.context_manager ]; then
			touch ../data/$1/.context_manager
		fi
		datetime="$(date)"
		echo "new data ($datetime)" >> ../data/$1/.context_manager
	fi
}

toggle_context_managers dependent
toggle_context_managers independent
toggle_context_managers zero-shot
toggle_context_managers processed_data
