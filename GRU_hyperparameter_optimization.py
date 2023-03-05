import config as conf
import functions.initialization_functions as init_funcs
import functions.GRU_functions as funcs

import os
#import logging
import shelve
import numpy as np
import pandas as pd
import multiprocessing as mp

if __name__ == '__main__':

    # Instantiate paths and params
    paths = conf.DataFilePaths()
    params = conf.GRU_model_parameters()

    # Get column index for parsed data
    index = shelve.open(paths.PARSED_DATA_COLUMN_INDEX)

    # Fire up logger
    logger = init_funcs.start_logger(
        logfile = f'{paths.LOG_DIR}/{params.log_file_name}',
        logstart_msg = 'Starting GRU hyperparameter optimization run.'
    )

    # Build parameter sets
    run_parameter_sets = []

    for block_size in params.block_sizes:
        for GRU_units in params.GRU_unit_nums:
            for learning_rate in params.learning_rates:
                for iteration in range(params.iterations):

                    run_parameter_sets.append([
                        iteration,
                        learning_rate, 
                        GRU_units,
                        block_size
                    ])
        
    # Make empty dataframe to hold results

    column_names = [
        'GPU',
        'Run number',
        'Iteration',
        'Block size',
        'GRU units',
        'Learning rate',
        'GRU private SMAPE score',
        'Control private SMAPE score',
        'GRU public SMAPE score',
        'Control public SMAPE score'
    ]

    error_data_df = pd.DataFrame(columns = column_names)

    # Helper function to log results and update free gpu list
    def log_result(result):
        '''Takes return from worker process. Saves run data
        to dataframe and parquet. Updates the free GPU list.'''
        
        try:
            error_data_df.loc[len(error_data_df)] = result
            error_data_df.to_parquet(params.optimization_data_output_file)

            GPU = result[0]
            free_GPUs.append(GPU)

            print(f'\n\n Added gpu {GPU} back to free gpus: {free_GPUs}', end = '')
            
        except Exception as e:
            print(f'Caught exception from GPU while saving results')
            
            # This prints the type, value, and stack trace of the
            # current exception being handled.
            traceback.print_exc()

            print()
            raise e

    # Main loop to submit jobs
    run_num = 0
    free_GPUs = [0, 1, 2, 3]
    total_runs = len(run_parameter_sets)

    # Instantiate pool
    pool = mp.Pool(processes = params.num_GPUs)

    # Loop on the parameter sets
    while True:

        if len(free_GPUs) > 0:
            gpu = free_GPUs.pop(0)

            # Get and unpack parameter set
            run_parameter_set = run_parameter_sets.pop(0)
            iteration = run_parameter_set[0]
            learning_rate = run_parameter_set[1]
            GRU_units = run_parameter_set[2]
            block_size = run_parameter_set[3]

            # Load and prep data
            input_file = f'{paths.PARSED_DATA_PATH}/{params.input_file_root_name}{params.block_size}.npy'
            timepoints = np.load(input_file)

            datasets = funcs.training_validation_testing_split(
                index,
                timepoints,
                num_counties = params.num_counties,
                input_data_type = params.input_data_type,
                testing_timepoints = params.testing_timepoints,
                training_split_fraction = params.training_split_fraction,
                pad_validation_data = params.pad_validation_data,
                forecast_horizon = params.forecast_horizon
            )

            datasets, training_mean, training_deviation = funcs.standardize_datasets(datasets)
            datasets = funcs.make_batch_major(datasets)

            result = pool.apply_async(funcs.train_GRU,
                args = (
                    gpu,
                    run_num,
                    total_runs,
                    iteration,
                    datasets,
                    block_size,
                    params.forecast_horizon,
                    training_mean, 
                    training_deviation,
                    params.epochs,
                    GRU_units,
                    learning_rate,
                    params.save_tensorboard_log,
                    params.tensorboard_log_dir,
                    params.tensorboard_histogram_freq,
                    params.save_model_checkpoints,
                    params.model_checkpoint_dir,
                    params.model_checkpoint_threshold,
                    params.model_checkpoint_variable,
                    params.early_stopping,
                    params.early_stopping_monitor,
                    params.early_stopping_min_delta,
                    params.early_stopping_patience,
                    params.verbose
                ),

                callback = log_result
            )

            run_num += 1

    # Clean up
    pool.close()
    pool.join()

    print('\n')