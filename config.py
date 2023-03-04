import os
import logging
import multiprocessing as mp

'''Configuration file for hardcoding python variables.
Used to store things like file paths, model hyperparameters etc.'''

PROJECT_NAME = 'godaddy-microbusiness-density-forecasting'

# Get path to this config file so that we can define
# other paths relative to it
PROJECT_ROOT_PATH = os.path.dirname(os.path.realpath(__file__))

# Logging stuff
LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(name)s:%(levelname)s - %(message)s'

class DataFilePaths:

    # Logs
    LOG_DIR = f'{PROJECT_ROOT_PATH}/logs'
    TENSORBOARD_LOGS = f'{LOG_DIR}/tensorboard'
    MODEL_CHECKPOINTS = f'{LOG_DIR}/model_checkpoints'

    # Data related files & paths
    DATA_PATH = f'{PROJECT_ROOT_PATH}/data'

    # Data input from Kaggle and/or other sources
    DATA_SOURCES_PATH = f'{DATA_PATH}/data_sources'
    KAGGLE_DATA_PATH = f'{DATA_SOURCES_PATH}/kaggle'
    CENSUS_DATA_PATH = f'{DATA_SOURCES_PATH}/census'

    # Parsed/formatted data for benchmarking, training and cross validation
    PARSED_DATA_PATH = f'{DATA_PATH}/parsed_data'
    PARSED_DATA_COLUMN_INDEX = f'{PARSED_DATA_PATH}/column_index'

    # Contest submission files
    SUBMISSIONS_PATH = f'{DATA_PATH}/submissions'

    # Leaderboard scoring test submission files
    LEADERBOARD_TEST_PATH = f'{SUBMISSIONS_PATH}/leaderboard_test'

    # Baseline benchmarking submission files
    BENCHMARKING_PATH = f'{SUBMISSIONS_PATH}/benchmarking'

    # Bootstrapping results
    BOOTSTRAPPING_RESULTS_PATH = f'{DATA_PATH}/bootstrapping_results'

# Linear model bootstrapping parameters

class LinearModelsBootstrappingParameters:

    # Run specific files
    log_file_name = 'linear_models_bootstrapping.log'
    input_file_root_name = 'updated_structured_bootstrap_blocksize'
    output_file_root_name = 'linear_models'

    # Experiment parameters
    num_samples = 180
    sample_size = 3000
    model_orders = [3,6,9,18]
    model_types = ['OLS', 'TS', 'Seigel', 'Ridge']
    time_fits = False

    n_cpus = mp.cpu_count() - 2
    samples_per_cpu = int(num_samples / n_cpus)

class ARIMA_model_parameters:

    # Run specific files
    log_file_name = 'ARIMA_hyperparameter_bootstrapping-block_size.log'
    input_file_root_name = 'updated_structured_bootstrap_blocksize'
    output_file_root_name = 'ARIMA_hyperparameter_bootstrapping-block_size'

    # Experiment parameters
    data_type = 'microbusiness_density'
    num_samples = 1800
    sample_size = 1000

    block_sizes = [8,16,32]
    lag_orders = [0,1,2,3,4]
    difference_degrees = [0,1,2,3]
    moving_average_orders = [0]

    # Parallelization stuff
    n_cpus = mp.cpu_count() - 2
    samples_per_cpu = int(num_samples / n_cpus)
    time_fits = False
    suppress_fit_warnings = True

class GRU_model_parameters():

    # Run specific files
    log_file_name = 'GRU_hyperparameter_optimization.log'
    input_file_root_name = 'updated_structured_bootstrap_blocksize'
    output_file_root_name = 'GRU_hyperparameter_optimization'

    # Data related stuff
    input_data_type = 'microbusiness_density'
    block_size = 13
    forecast_horizon = 5
    num_counties = 'all'
    testing_timepoints = None
    training_split_fraction = 0.7
    pad_validation_data = True

    # Run options
    verbose = 0

    save_tensorboard_log = True
    tensorboard_histogram_freq = 1

    save_model_checkpoints = True
    model_checkpoint_threshold = 0.1,
    model_checkpoint_variable = 'val_MAE',

    early_stopping = True
    early_stopping_monitor = 'val_MAE'
    early_stopping_min_delta = 0.01
    early_stopping_patience = 2

    # Hyperparameters
    GRU_units = 8
    learning_rate = 0.0002
    epochs = 100
