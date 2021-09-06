import logging
import os
import csv
from utils.utils import get_configs

level_dict = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING
}

config_path = "config/debug_configs.json"
configs = get_configs(config_path)
INIT = False


def get_event_logger():
    global INIT
    if INIT:
        return logging.getLogger('root.yuanhang')

    if configs.exam_empty:
        exc = "Event log file is NOT empty! Quit to avoid overwriting"
        assert (not os.path.exists(configs.event_file)) or \
               os.path.getsize(configs.event_file) == 0, exc

    logger = logging.getLogger('root.yuanhang')
    logger.setLevel(level_dict[configs.level])

    fmt = logging.Formatter(fmt=configs.format)
    fh = logging.FileHandler(configs.event_file, 'w')
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    if configs.level == 'debug':
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    INIT = True
    return logger


class DataLogger:
    def __init__(self, fieldnames):
        if configs.exam_empty:
            exc = "Event log file is NOT empty! Quit to avoid overwriting"
            assert (not os.path.exists(configs.event_file)) or \
                   os.path.getsize(configs.event_file) == 0, exc
        f = open(configs.data_file, 'w', newline='')
        self.writer = csv.DictWriter(f, fieldnames)
        self.writer.writeheader()

    def log_data(self, d: dict):
        self.writer.writerow(d)
