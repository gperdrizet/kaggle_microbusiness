import config
import logging

from logging.handlers import TimedRotatingFileHandler
from multiprocessing_logging import install_mp_handler

def start_logger(
    logfile = f'{config.LOG_DIR}/unspecified.log',
    logstart_msg = 'Log start message not set.'
):
    '''Sets up logger for run of anything using logging. Takes logfile
    name and log startup message string. Also reads a bunch of stuff
    from config.py. Returns logger.'''
    
    logger = logging.getLogger()
    logger.setLevel(config.LOG_LEVEL)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    
    handler = TimedRotatingFileHandler(
        logfile,
        when=config.LOG_ROTATION_FREQUENCY,
        interval=1,
        backupCount=config.LOG_BACKUP_COUNT
    )

    logFormatter = logging.Formatter(config.LOG_FORMAT)
    handler.setFormatter(logFormatter)
    handler.setLevel(config.LOG_LEVEL)
    logger.addHandler(handler)
    install_mp_handler(logger)

    logger.info(f'############################################################')
    logger.debug('This is a debug message')
    logger.info('This is an info message')
    logger.warning('This is a warn message')
    logger.error('This is an error message')
    logger.critical('This is a critical message')
    logger.info(f'####### {logstart_msg} #######')

    return logger