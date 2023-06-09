executable            = scripts/baselines.sh
output                = logs/baselines.out
error                 = logs/baselines.log
log                   = logs/baselines.info
arguments             = "-d ../data/processed_data/baselines_dys --output_dir models/baselines"
transfer_executable   = False
max_retries           = 2
retry_until           = !member( ExitCode, {130})
request_GPUs          = 1
job_lease_duration    = 3600
request_memory        = 8*1024
stream_output         = True
notification          = complete
notify_user           = hsteinm@uw.edu
queue
