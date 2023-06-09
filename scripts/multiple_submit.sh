#!/bin/sh

# MAKE SURE TO USE IN TMUX!

function cmd_submit() {
    condor_submit cmd/train.cmd -append "queue JOB in ($1)" && \
    echo "Waiting for $1 to finish"
    condor_wait "results/logs/$1.info"
    echo "Done with job $1"
}

for job in "$@"; do
    cmd_submit $job
done
echo "All jobs complete!"
DATE=$(date)
git add -A && git commit -m "(auto commit $DATE) new results " && git push