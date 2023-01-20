import os

'''Configuration file for hardcoding python variables.
Used to store things like file paths, model hyperparameters etc.'''

PROJECT_NAME = 'godaddy-microbusiness-density-forecasting'

# Get path to this config file so that we can define 
# other paths relative to it
PROJECT_ROOT_PATH = os.path.dirname(os.path.realpath(__file__))

# Data related files & paths
DATA_PATH = f'{PROJECT_ROOT_PATH}/data/'