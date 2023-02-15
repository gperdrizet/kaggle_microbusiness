import config
import logging

from multiprocessing_logging import install_mp_handler

def start_logger(
    logfile = 'unspecified.log',
    logstart_msg = 'Log start message not set.'
):
    '''Sets up logger for run of anything using logging. Takes logfile
    name and log startup message string. Also reads a bunch of stuff
    from config.py. Returns logger.'''
    
    logger = logging.getLogger()
    logger.setLevel(config.LOG_LEVEL)
    
    handler = logging.FileHandler(
        logfile,
        mode='w'
    )

    logFormatter = logging.Formatter(config.LOG_FORMAT)
    handler.setFormatter(logFormatter)
    handler.setLevel(config.LOG_LEVEL)
    logger.addHandler(handler)
    install_mp_handler(logger)

    logger.info(f'####### {logstart_msg} #######')

    return logger