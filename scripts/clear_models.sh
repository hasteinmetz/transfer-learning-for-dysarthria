#!/bin/bash

PWD=$(pwd)

if [[ $(basename $PWD) == "scripts" ]]; then
    cd ..
fi

echo "Remove:"

for dir in models/*; do
    name=${dir#*/}
    if [ -d "$dir" ] && [[ ! "$dir" =~ "tmp" ]] && [[ "$@" =~ "$name" ]]; then
        for file in $dir/*; do
            if [[ "$file" =~ "pytorch_model.bin" ]]; then
                printf "  - $file [y/n]: "
                read varname
                if [[ "$varname" == "y" ]]; then 
                    rm file
                    printf "\n"
                elif [[ "$varname" != "n" ]]; then
                    printf "\nUnrecognized input. Exiting.\n"
                    exit 1
                fi
            fi
            if [[ "$file" =~ "checkpoint" ]]; then
                printf "  - $file\n"
                read varname
                if [[ "$varname" == "y" ]]; then 
                    rm -rf file
                    printf "\n"
                elif [[ "$varname" != "n" ]]; then
                    printf "\nUnrecognized input. Exiting.\n"
                    exit 1
                fi
            fi
        done
    fi
done
exit 0 