import logging
import os
import csv
from utils.utils import get_configs

level_dict = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING
}

config_path = "config/logger.json"
configs = get_configs(config_path)


def init_event_logger():
    if configs.exam_empty:
        assert os.path.getsize(configs.event_log) == 0, "Event log file is NOT empty! Quit to avoid overwriting"
    logger = logging.getLogger()
    logger.setLevel(level_dict[configs.level])

    fmt = logging.Formatter(fmt=configs.format)
    fh = logging.FileHandler(configs.event_file, 'w')
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    if configs.level == 'debug':
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)


class DataLogger:
    def __init__(self, fieldnames):
        if configs.exam_empty:
            assert os.path.getsize(configs.data_log) == 0, "Event log file is NOT empty! Quit to avoid overwriting"
        f = open(configs.data_file, 'w', newline='')
        self.writer = csv.DictWriter(f, fieldnames)

    def log_data(self, d: dict):
        self.writer.writerow({k: d[k][-1]} for k in d.keys())
