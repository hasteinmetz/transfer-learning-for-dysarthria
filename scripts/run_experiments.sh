#!/bin/sh

source ~/.bashrc

if [ $# -ge 1 ]; then
    echo "The script does NOT take arguments. Arguments are constructed in the script itself."
    exit 1
fi

EXPERIMENTS=(
    # "dependent/base"
    # "dependent/finetune"
    # "dependent/finetune-dys"
    # "dependent/multitask"
    # "dependent/multitask-same"
    # "independent/base"
    # "independent/finetune-dys"
    "independent/finetune"
    "independent/multitask"
    # "zero-shot/base"
    # "zero-shot/finetune"
    # "zero-shot/finetune-dys"
    # "zero-shot/multitask-same"
    # "zero-shot/multitask"
);

BRANCHNAME=""

function get_branch_name() {
    BRANCHNAME=$(git symbolic-ref -q HEAD)
    BRANCHNAME=${BRANCHNAME##refs/heads/}
    BRANCHNAME=${BRANCHNAME:-HEAD}
}

# MAKE SURE TO USE IN TMUX!

function cmd_submit() {
    DATE=$(date '+%D %H:%M')
    echo "($DATE) Submitting job $1 and waiting for completion..."
    QUEUE="queue JOB in ($1)"
    condor_submit cmd/train-quatro.cmd -append "$QUEUE" && condor_sync "logs/$1.info"
    ecode=$?
    if [ $ecode -ne 0 ]; then
        DATE=$(date '+%D %H:%M')
        echo "($DATE) Could not complete job $1 :("
        exit 1
    else
        DATE=$(date '+%D %H:%M')
        echo "($DATE) Done with job $1!"
        get_branch_name
        if [ $BRANCHNAME == 'main' ]; then
            git add results && git commit -m "($DATE) $1 results" && \
            git push
        fi
    fi
}

for job in ${EXPERIMENTS[@]}; do
    cmd_submit $job
done
echo "Done all experiments!"
