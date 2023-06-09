#!/bin/sh

JOB=$1
shift
ARGS="$@"
QUATRO="N"
CPU="N"

for arg in $ARGS; do
    shift
    if [[ "$arg" =~ "--quatro" ]]; then
        QUATRO="Y"
    elif [[ "$arg" =~ "--cpu" ]]; then
        CPU="Y"
    else
        set -- "$@" "$arg"
    fi
done

if [ $QUATRO == "Y" ]; then
    condor_submit cmd/train-quatro.cmd -append "queue JOB, ARGS from ($JOB, $@)"
elif [ $CPU == "Y" ]; then
    condor_submit cmd/train-cpu.cmd -append "queue JOB, ARGS from ($JOB, $@)"
else
    PIPEARGS="queue JOB, ARGS from ($JOB, $@)"
    condor_submit cmd/train.cmd -append "$PIPEARGS"
fi