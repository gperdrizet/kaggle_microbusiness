import os
import logging

'''Configuration file for hardcoding python variables.
Used to store things like file paths, model hyperparameters etc.'''

PROJECT_NAME = 'godaddy-microbusiness-density-forecasting'

# Get path to this config file so that we can define 
# other paths relative to it
PROJECT_ROOT_PATH = os.path.dirname(os.path.realpath(__file__))

# Logging stuff
LOG_LEVEL = logging.INFO
LOG_ROTATION_FREQUENCY = 'h'
LOG_BACKUP_COUNT = 24
LOG_FORMAT = '%(asctime)s:%(name)s:%(levelname)s - %(message)s'
LOG_DIR = f'{PROJECT_ROOT_PATH}/logs/'
#DATETIME_FORMAT = '%Y-%m-%dT%H:%M:%SZ'

# Data related files & paths
DATA_PATH = f'{PROJECT_ROOT_PATH}/data'

# Data input from Kaggle and/or other sources
DATA_SOURCES_PATH = f'{DATA_PATH}/data_sources'
KAGGLE_DATA_PATH = f'{DATA_SOURCES_PATH}/kaggle'

# Parsed/formatted data for benchmarking, training and cross validation
PARSED_DATA_PATH = f'{DATA_PATH}/parsed_data'

# Contest submission files
SUBMISSIONS_PATH = f'{DATA_PATH}/submissions'

# Leaderboard scoring test submission files
LEADERBOARD_TEST_PATH = f'{SUBMISSIONS_PATH}/leaderboard_test'

# Baseline benchmarking submission files
BENCHMARKING_PATH = f'{SUBMISSIONS_PATH}/benchmarking'