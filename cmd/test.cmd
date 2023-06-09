executable            = scripts/train.sh 
output                = logs/$(JOB).out
error                 = logs/$(JOB).log
log                   = logs/$(JOB).info
arguments             = "-c configs/$(JOB).conf $(CPU) $(ARGS)"
transfer_executable   = False
max_retries           = 2
retry_until           = !member( ExitCode, {130})
job_lease_duration    = 3600
request_memory        = 8*1024
stream_output         = True

if defined request_GPUs
   CPU = 
else
   CPU = --cpu
endif
