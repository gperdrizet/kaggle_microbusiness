import config
import logging
import numpy as np

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

def load_inspect_parsed_data(
        input_file = None, 
        params = None
    ):
    
    timepoints = np.load(input_file)

    # Log some details about the input data & run parameters
    logging.info('')
    logging.info(f'CPUs: {params.n_cpus}')
    logging.info(f'Samples: {params.num_samples} ({params.samples_per_cpu} per CPU)')
    logging.info(f'Sample size: {params.sample_size}')
    logging.info('')
    logging.info(f'Input timepoints shape: {timepoints.shape}')
    logging.info('')
    logging.info('Input column types:')

    for column in timepoints[0,0,0,0:]:
        logging.info(f'{type(column)}')

    logging.info('')

    logging.info(f'Example input block:')

    for row in timepoints[0,0,0:,]:
        row = [f'{x:.2e}' for x in row]
        logging.info(f'{row}')

    logging.info('')

    return timepoints