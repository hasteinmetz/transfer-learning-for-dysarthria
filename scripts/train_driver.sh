#!/bin/bash

echo "Using python $(which python)"

if [[ "$OSTYPE" == "darwin"* ]]; then
    if [[ "$@" =~  "--cpu" ]]; then
        echo "On M1 Mac OS. Running: python src/train.py $@"
        python src/train.py $@
        exitcode=$?
        exit $exitcode
    else
        echo "On M1 Mac OS. Running: accelerate launch src/train.py $@"
        PYTORCH_ENABLE_MPS_FALLBACK=1 accelerate launch src/train.py $@
        exitcode=$?
        exit $exitcode
    fi
elif [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    if [[ "$@" =~ "--cpu" ]]; then
        echo "Not using GPU. Running: python src/train.py $@"
    else
        echo "No GPUs found. Running: python src/train.py $@"
    fi
    python src/train.py $@
    exitcode=$?
    exit $exitcode
else
    IFS=","
    read -a gpus <<< "$CUDA_VISIBLE_DEVICES"
    NUM_GPUS=${#gpus[*]}
    if [[ "$@" =~ "--deepspeed" ]]; then
        for arg in "$@"; do
            shift
            if [ "$arg" != "--deepspeed" ]; then
                set -- "$@" "$arg"
            fi
        done
        echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES deepspeed src/train.py $@"
        echo " --deepspeed configs/deepspeed/ds_config.json"
        CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES deepspeed src/train.py $@ \
            --deepspeed configs/deepspeed/ds_config.json
    else
        if [ $NUM_GPUS -gt 1 ]; then  
            if [[ "$@" =~ "--fsdp" ]]; then
                echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun src/train.py $@"
                echo "  --nproc_per_node $NUM_GPUS --log_level_replica error --log_on_each_node 0"
                TMPDIR="/projects/assigned/2223_hsteinm/.cache/huggingface/tmp" \
                    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
                    torchrun src/train.py $@ --nproc_per_node $NUM_GPUS \
                        --log_level_replica error --log_on_each_node 0
            else
                echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python src/train.py $@"
                echo "  --log_level_replica error --log_on_each_node 0"
                TMPDIR="/projects/assigned/2223_hsteinm/.cache/huggingface/tmp" \
                    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
                    python src/train.py $@ --log_level_replica error --log_on_each_node 0
            fi
        else
            echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python src/train.py $@"
            TMPDIR="/projects/assigned/2223_hsteinm/.cache/huggingface/tmp" \
                CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python src/train.py $@
        fi
    fi
    exitcode=$?
    exit $exitcode
fi
