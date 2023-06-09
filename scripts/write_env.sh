#!/bin/bash

if [[ "$OSTYPE" == "darwin"* ]]; then
    ~/Documents/UW/thesis/repo/scripts/check_dir.sh
else
    ~/thesis/repo/scripts/check_dir.sh
fi

echo "Saving $1 file. NOTE: be sure to specify the operating system in the file name!"
if [ -d ~/anaconda3 ];
    then CONDADIR="anaconda3"
elif [ -d ~/opt/anaconda3 ];
    then CONDADIR="opt/anaconda3"
else
    echo "Can't find Conda directory. Please edit script."
    exit 1
fi
source ~/$CONDADIR/etc/profile.d/conda.sh
conda activate thesis-env 
conda env export --channel pytorch --channel conda-forge > $1 
NEWFILE=$( head -n $(expr $(cat $1 | wc -l) - 1) $1 )
echo "$NEWFILE" > $1
exit 0