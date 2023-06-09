executable            = scripts/train.sh 
output                = logs/$(JOB).out
error                 = logs/$(JOB).log
log                   = logs/$(JOB).info
arguments             = "-c configs/$(JOB).conf $(CPU) $(ARGS)"
transfer_executable   = False
max_retries           = 2
retry_until           = !member( ExitCode, {130})
job_lease_duration    = 4192
request_memory        = 8*1024
stream_output         = True
nice_user             = True

if defined request_GPUs
   CPU = 
else
   CPU = --cpu
endif
