executable            = scripts/train.sh 
output                = logs/debug.out
error                 = logs/debug.log
log                   = logs/debug.info
arguments             = "-c configs/debug.conf $(CPU)"
transfer_executable   = False
request_GPUs          = 1
retry_until           = !member( ExitCode, {130})
request_memory        = 4*1024
stream_output         = True

if defined request_GPUs
   CPU = 
else
   CPU = --cpu
endif

queue
