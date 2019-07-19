import os

import logging
from logging import Formatter
from logging.handlers import RotatingFileHandler

import json
from easydict import EasyDict
from pprint import pprint
from datetime import datetime

from utils.dirs import create_dirs



def setup_logging(config):
    log_dir = config.LOG_DIR
    create_dirs([log_dir, config.SUMMARY_DIR, config.CHECKPOINT_DIR, config.OUTPUT_DIR])

    log_file_format = "%(asctime)s : %(message)s"
    log_console_format = "%(message)s"

    # Main logger
    main_logger = logging.getLogger()
    main_logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(Formatter(log_console_format))

    exp_file_handler = RotatingFileHandler(
        os.path.join(log_dir, '{:%Y-%m-%d-%H-%M-%S}.log'.format(datetime.now())), 
        maxBytes=10**6, 
        backupCount=5)
    exp_file_handler.setLevel(logging.DEBUG)
    exp_file_handler.setFormatter(Formatter(log_file_format))

    main_logger.addHandler(console_handler)
    main_logger.addHandler(exp_file_handler)




