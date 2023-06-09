#!/bin/bash

source ~/.bashrc

if [[ "$(basename $(pwd))" != "repo" ]]; then
    echo "Moving to ~/thesis/repo"
    cd $HOME/thesis/repo
fi

ecode=$?
if [ $ecode -ne 0 ]; then
    echo "(check_dir code: $ecode) Issue with the source command..."
fi

if [ "$CONDA_DEFAULT_ENV" != "thesis-env" ]; then
    echo "Activating thesis-env"
    timeout --preserve-status 120 conda activate thesis-env
    ecode=$?
    if [ $ecode -ne 0 ]; then
        echo "(check_dir code: $ecode) Issue loading conda environment. Aborting script."
    else
        echo "(check_dir code: $ecode) Successfully loaded conda environment: $CONDA_DEFAULT_ENV"
        exit $ecode
    fi
else
    echo "(check_dir code: $ecode) Successfully loaded conda environment: $CONDA_DEFAULT_ENV"
    exit $ecode
fi