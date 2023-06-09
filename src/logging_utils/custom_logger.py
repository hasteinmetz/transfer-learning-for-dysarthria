#!/bin/env python

import logging
from datasets.utils import logging as ds_logging
from transformers.utils import logging as tf_logging
from numpy import warnings as np_warnings
from numpy import VisibleDeprecationWarning

import sys
import os
import re
import json
from typing import *


''' ##### Get globals from logger_variables.json ##### '''
data_file = os.path.join(os.path.dirname(__file__), 'logger_variables.json')
with open(data_file, 'r') as variables:
    LOGGER_VARS = json.load(variables)
''' ################################################## '''


logging.basicConfig(
    format=LOGGER_VARS['DEFAULT_FMT'],
    datefmt=LOGGER_VARS['DATE_FMT'],
    level=logging.INFO,
    handlers=[]
)


def _level_proc_log(levelNum):
    def _proc_log(self, message, *args, **kws):
        if self.isEnabledFor(levelNum):
            self._log(levelNum, message, args, **kws)
    return _proc_log


def _level_root_proc_log(levelNum):
    def _root_proc_log(message, *args, **kwargs):
        logging.log(levelNum, message, *args, **kwargs)
    return _root_proc_log


def add_new_logging_levels(
        levels=[
            ("MESSAGE", LOGGER_VARS['MESSAGE']), 
            ("WATCH_OUT", LOGGER_VARS['WATCH_OUT'])
        ]
    ):
    # From answers on: https://stackoverflow.com/q/2183233/
    for (levelName, levelNum) in levels:
        method_name = levelName.lower()
        _log_proc_for_lvl = _level_proc_log(levelNum)
        _root_log_proc_for_lvl = _level_root_proc_log(levelNum)
        logging.addLevelName(levelNum, levelName)
        setattr(logging, levelName, levelNum)
        setattr(logging.getLoggerClass(), method_name, _log_proc_for_lvl)
        setattr(logging, method_name, _root_log_proc_for_lvl)


class LogFormatter(logging.Formatter):
    '''CODE adapted from: https://stackoverflow.com/q/1343227'''
    
    err_fmt  = "(ERROR|%(asctime)s|%(name)s|%(filename)s|%(lineno)s) >> %(msg)s"
    warning_fmt  = "(WARNING|%(asctime)s|%(name)s|%(filename)s) >> %(msg)s"
    critical_fmt  = "(WATCH OUT|%(asctime)s|%(name)s|%(filename)s) >> %(msg)s"
    dbg_fmt  = "(DEBUG|%(asctime)s|%(name)s|%(filename)s) >> %(msg)s"
    info_fmt = "(INFO|%(asctime)s|%(filename)s) %(msg)s"
    process_fmt = "%(msg)s"

    color = dict(
        grey = "\x1b[38;21m",
        yellow = "\x1b[33;21m",
        red = "\x1b[31;21m",
        bold_red = "\x1b[31;1m",
        green = "\033[32m",
        reset = "\x1b[0m",
    )

    reset = color.get('reset')

    def __init__(self, colors: Optional[bool] = None):
        '''Colors is automatically determined by whether text is piped or not'''
        if colors is None:
            self.colors_toggle = sys.stdout.isatty()
        else:
            self.colors_toggle = colors
        super().__init__(
            fmt=LOGGER_VARS['DEFAULT_FMT'], 
            datefmt=LOGGER_VARS['DATE_FMT']
        ) 

    def format(self, record: logging.LogRecord):

        if hasattr(record, 'no_format') and record.no_format:
            # Code adapted from: https://stackoverflow.com/q/34954373/
            return record.getMessage()

        # Save the original format configured by the user
        # when the logger formatter was instantiated
        format_orig = self._style._fmt

        def _format(fmt: str, color: Optional[str] = None):
            if color is not None and self.colors_toggle:
                assert color in self.color
                return LogFormatter.color.get(color) + fmt + LogFormatter.reset
            else:
                return fmt

        # Replace the original format with one customized by logging level
        if record.levelno == logging.DEBUG:
            self._style._fmt = _format(fmt=LogFormatter.dbg_fmt, color='grey')

        elif record.levelno == logging.INFO:
            self._style._fmt = _format(fmt=LogFormatter.info_fmt, color='grey')

        elif record.levelno == logging.WARNING:
            self._style._fmt = _format(fmt=LogFormatter.warning_fmt, color='yellow')
            
        elif record.levelno == LOGGER_VARS['MESSAGE']:
            self._style._fmt = _format(fmt=LogFormatter.process_fmt, color='green')

        elif record.levelno == LOGGER_VARS['WATCH_OUT']:
            self._style._fmt = _format(fmt=LogFormatter.critical_fmt, color='red')

        elif record.levelno == logging.ERROR:
            self._style._fmt = _format(fmt=LogFormatter.err_fmt, color='bold_red')

        # Call the original formatter class to do the grunt work
        result = super().format(record)

        # Preprocess message to remove multiline strings and replace with single line
        result = re.sub(r'(?<=[\.,a-zA-Z])\s*\n\s*(?=[a-zA-Z])', ' ', result)

        # Restore the original format configured by the user
        self._fmt = format_orig

        return result


def set_up_logging(args: "Arguments"):
    '''Set up the loggers according to config file'''

    add_new_logging_levels()

    logger = logging.getLogger("wav2vec2_logger")

    fmt = LogFormatter()

    log_level = args.get_log_levels(
        log_level_dict=logging._nameToLevel
    )
    if log_level % 10 != 0:
        log_level = logging.MESSAGE if log_level == 25 else logging.WATCH_OUT
    else:
        log_level = log_level

    _ds_log = ds_logging._get_library_root_logger

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)

    error_handler = logging.StreamHandler(sys.stderr)
    error_handler.setFormatter(fmt)

    if not sys.stdout.isatty():
        # if printing to a file, then print stderr at info level

        console_handler.setLevel(log_level)
        error_handler.setLevel(logging.INFO)

        _ds_log().setLevel(logging.INFO)

        os.environ["TRANSFORMERS_VERBOSITY"] = "info"
        tf_logging.set_verbosity(logging.INFO)
    else:
        # print all loggers at log_level
        console_handler.setLevel(log_level)
        error_handler.setLevel(logging.CRITICAL)

        if log_level == logging.MESSAGE or log_level == logging.WATCH_OUT:
            hf_log_level = logging.WARNING
        else:
            hf_log_level = log_level

        _ds_log().setLevel(hf_log_level)

        os.environ["TRANSFORMERS_VERBOSITY"] = logging._levelToName[hf_log_level].lower()
        tf_logging.set_verbosity(hf_log_level)

    _ds_log().handlers.clear()
    _ds_log().addHandler(console_handler)
    _ds_log().addHandler(error_handler)

    tf_logging.disable_default_handler()
    tf_logging.add_handler(console_handler)
    tf_logging.add_handler(error_handler)

    logger.handlers.clear()
    logger.addHandler(console_handler)
    logger.addHandler(error_handler)
    logger.propagate = False

    np_warnings.filterwarnings('ignore', category=VisibleDeprecationWarning)

    return logger


if __name__ == '__main__':
    pass