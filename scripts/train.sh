#!/bin/bash

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

# go to the write directory
if [[ "$(basename $(pwd))" != "repo" ]]; then
	echo "Moving to ~/thesis/repo"
	cd ~/thesis/repo
fi

# Main driver for training
scripts/train_driver.sh $@
exitcode=$?
echo "Exiting with status: $exitcode"
exit $exitcode