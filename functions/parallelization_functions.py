# Add parent directory to path to allow import of config.py
import sys
sys.path.append('..')
import functions.bootstrapping_functions as bootstrap_funcs

import logging
import numpy as np
import pandas as pd
import multiprocessing as mp

def start_multiprocessing_pool():
    # Instantiate multiprocessing pool to parallelize over folds
    n_cpus = mp.cpu_count() - 2

    logging.info('')
    logging.info(f'Starting processes for {n_cpus} CPUs (available - 2)')

    pool = mp.Pool(processes = n_cpus)

    # Holder for result objects
    result_objects = []

    return pool, result_objects

def parallel_bootstrapped_smape(
    timepoints, 
    sample_num, 
    sample_size, 
    model_orders, 
    model_types,
    time_fits = False
):
    
    # Holder for sample results
    data = {
        'sample': [],
        'model_type': [],
        'model_order': [],
        'SMAPE_values': [],
        'detrended_SMAPE_values': [],
        'MBD_predictions': [],
        'detrended_MBD_predictions': [],
        'MBD_inputs': [],
        'detrended_MBD_inputs': [],
        'MBD_actual': []
    }

    # Loop on model orders
    for model_order in model_orders:
        result = bootstrap_funcs.bootstrap_smape_scores(            
            timepoints, 
            sample_num, 
            sample_size, 
            model_order, 
            model_types,
            time_fits
        )

        # Add results for this order
        for key, value in result.items():
            data[key].extend(value)

    return data

def parallel_ARIMA_gridsearch(
    timepoints, 
    sample_num,
    sample_size,
    lag_orders,
    difference_degrees,
    moving_average_orders, 
    time_fits = False
):
    
    # Holder for sample results
    data = {
        'sample': [],
        'model_type': [],
        'lag_order': [],
        'difference_degree': [],
        'moving_average_order': [],
        'SMAPE_value': [],
        'MBD_prediction': [],
        'MBD_inputs': [],
        'MBD_actual': []
    }

    # Loop on model orders
    for lag_order in lag_orders:
        for difference_degree in difference_degrees:
            for moving_average_order in moving_average_orders:

                result = bootstrap_funcs.bootstrap_ARIMA_smape_scores(            
                    timepoints, 
                    sample_num, 
                    sample_size, 
                    lag_order,
                    difference_degree,
                    moving_average_order,
                    time_fits
                )

                # Add results for this order
                for key, value in result.items():
                    data[key].extend(value)

    return data

def cleanup_bootstrapping_multiprocessing_pool(pool, result_objects):

    # Collect results
    results = [result.get() for result in result_objects]

    # Holder for parsed sample results
    data = {
        'sample': [],
        'model_type': [],
        'model_order': [],
        'SMAPE_values': [],
        'detrended_SMAPE_values': [],
        'MBD_predictions': [],
        'detrended_MBD_predictions': [],
        'MBD_inputs': [],
        'detrended_MBD_inputs': [],
        'MBD_actual': []
    }

    for result in results:
        for key, value in result.items():
            data[key].extend(value)

    # Clean up
    pool.close()
    pool.join()

    return data

def cleanup_ARIMA_bootstrapping_multiprocessing_pool(pool, result_objects):

    # Collect results
    results = [result.get() for result in result_objects]

    # Holder for parsed sample results
    data = {
        'sample': [],
        'model_type': [],
        'lag_order': [],
        'difference_degree': [],
        'moving_average_order': [],
        'SMAPE_value': [],
        'MBD_prediction': [],
        'MBD_inputs': [],
        'MBD_actual': []
    }

    for result in results:
        for key, value in result.items():
            data[key].extend(value)

    # Clean up
    pool.close()
    pool.join()

    return data