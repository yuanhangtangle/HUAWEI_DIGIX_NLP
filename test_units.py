import logging
from utils.logger import get_event_logger


def test_logging_system():
    logger = get_event_logger()
    logger.info('in test logging system')
    logger = get_event_logger()
    logger.info('another call')
    logging.info('by logging')


if __name__ == "__main__":
    test_logging_system()