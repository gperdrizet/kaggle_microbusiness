import config as conf
import functions.initialization_functions as init_funcs
import functions.GRU_functions as funcs

#import logging
import shelve
import traceback
import numpy as np
import pandas as pd
import multiprocessing as mp
#import tensorflow as tf

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

    run_parameter_sets, error_data_df = funcs.setup_results_output(
        params.optimization_data_output_file,
        params.hyperparameters,
        params.iterations
    )

    # Helper function to log results and update free gpu list
    def log_result(result):
        '''Takes return from worker process. Saves run data
        to dataframe and parquet. Updates the free GPU list.'''
        
        try:
            error_data_df.loc[len(error_data_df)] = result
            error_data_df.to_parquet(params.optimization_data_output_file)

            GPU = result[0]
            free_GPUs.append(GPU)
            
        except Exception as e:
            print(f'Caught exception from GPU while saving results')
            
            # This prints the type, value, and stack trace of the
            # current exception being handled.
            traceback.print_exc()

            print()
            raise e

    # Main loop to submit jobs
    run_num = 0

    # Set up initial list of free GPUs
    GPUs = [i for i in range(params.num_GPUs)]
    
    free_GPUs = []

    for i in range(params.jobs_per_GPU):
        free_GPUs.extend(GPUs)

    # Instantiate pool
    pool = mp.Pool(
        processes = (params.num_GPUs * params.jobs_per_GPU),
        maxtasksperchild = params.max_tasks_per_child
    )

    print('')
    print(f'{len(run_parameter_sets)} total parameter sets.')

    # Loop on the parameter sets
    while len(run_parameter_sets) > 0:

        if len(free_GPUs) > 0:

            GPU = free_GPUs.pop(0)

            # When starting a new run, print the number of parameter sets remaining
            print(f'{len(run_parameter_sets)} parameter sets remaining, submitting next job to GPU {GPU}')

            # Get and unpack parameter set
            run_parameter_set = run_parameter_sets.pop(0)
            block_size = run_parameter_set[0]
            GRU_units = run_parameter_set[1]
            learning_rate = run_parameter_set[2]
            iteration = run_parameter_set[3]

            result = pool.apply_async(funcs.train_GRU,
                args = (
                    GPU,
                    run_num,
                    iteration,
                    paths.PARSED_DATA_PATH,
                    params.input_file_root_name,
                    index,
                    params.num_counties,
                    params.input_data_type,
                    params.testing_timepoints,
                    params.training_split_fraction,
                    params.pad_validation_data,
                    block_size,
                    params.forecast_horizon,
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
