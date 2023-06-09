from .custom_logger import (
    set_up_logging,
    LOGGER_VARS
)
from .cuda_info import (
    query_cuda_memory,
    resolve_cuda,
    fix_batch_size
)

# def load_logging_variables():
#     import json
#     with open('logging_variables.json', 'r') as log_vars:
#         logging_vars = json.load(log_vars)
#     default_names = {
#         int(k): v for k, v in logging_vars["DEFAULT_LOG_NAMES"].items()
#     }
#     logging_vars["DEFAULT_LOG_NAMES"] = default_names
#     return logging_vars

# globals().update(load_logging_variables())


if __name__ == '__main__':
    pass