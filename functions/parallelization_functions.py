# Add parent directory to path to allow import of config.py
import sys
sys.path.append('..')
import functions.bootstrapping_functions as bootstrap_funcs

import logging
import numpy as np
import pandas as pd
import multiprocessing as mp

def start_multiprocessing_pool(n_cpus):
    # Instantiate multiprocessing pool to parallelize over folds
    logging.info('')
    logging.info(f'Starting processes for {n_cpus} CPUs.')

    pool = mp.Pool(processes = n_cpus)

    # Holder for result objects
    result_objects = []

    return pool, result_objects

def parallel_bootstrapped_linear_smape(
    index,
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
        'total_SMAPE_values': [],
        'public_SMAPE_value': [],
        'private_SMAPE_values': [],
        'detrended_total_SMAPE_values': [],
        'detrended_public_SMAPE_value': [],
        'detrended_private_SMAPE_values': [],
        'MBD_predictions': [],
        'detrended_MBD_predictions': [],
        'MBD_inputs': [],
        'detrended_MBD_inputs': [],
        'MBD_actual': []
    }

    # Loop on model orders
    for model_order in model_orders:
        result = bootstrap_funcs.bootstrap_linear_smape_scores(
            index,
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
    parsed_data_path,
    input_file_root_name,
    sample_num,
    sample_size,
    block_sizes,
    index,
    data_type,
    lag_orders,
    difference_degrees,
    moving_average_orders,
    suppress_fit_warnings = True,
    time_fits = False
):

    # Holder for sample results
    data = {
        'sample': [],
        'model_type': [],
        'block_size': [],
        'lag_order': [],
        'difference_degree': [],
        'moving_average_order': [],
        'SMAPE_value': [],
        'MBD_prediction': [],
        'MBD_inputs': [],
        'MBD_actual': [],
        'fit_residuals': [],
        'AIC': [],
        'BIC': [],
        'public_SMAPE': [],
        'private_SMAPE': []
    }

    # Loop on model parameters
    for block_size in block_sizes:

        # Load up parsed data for this blocksize
        input_file = f'{parsed_data_path}/{input_file_root_name}{block_size}.npy'
        logging.info(f'Input file: {input_file}')
        timepoints = np.load(input_file)

        for lag_order in lag_orders:
            for difference_degree in difference_degrees:
                for moving_average_order in moving_average_orders:

                    result = bootstrap_funcs.bootstrap_ARIMA_smape_scores(
                        timepoints,
                        sample_num,
                        sample_size,
                        index,
                        data_type,
                        lag_order,
                        difference_degree,
                        moving_average_order,
                        suppress_fit_warnings,
                        time_fits
                    )

                    # Add results for this order
                    for key, value in result.items():
                        data[key].extend(value)

                    # Add blocksize to results
                    data['block_size'].extend([block_size] * len(result['model_type']))

    return data

def cleanup_bootstrapping_multiprocessing_pool(pool, result_objects):

    # Collect results
    results = [result.get() for result in result_objects]

    # Holder for parsed sample results
    data = {
        'sample': [],
        'model_type': [],
        'model_order': [],
        'total_SMAPE_values': [],
        'public_SMAPE_value': [],
        'private_SMAPE_values': [],
        'detrended_total_SMAPE_values': [],
        'detrended_public_SMAPE_value': [],
        'detrended_private_SMAPE_values': [],
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
        'block_size': [],
        'lag_order': [],
        'difference_degree': [],
        'moving_average_order': [],
        'SMAPE_value': [],
        'MBD_prediction': [],
        'MBD_inputs': [],
        'MBD_actual': [],
        'fit_residuals': [],
        'AIC': [],
        'BIC': [],
        'public_SMAPE': [],
        'private_SMAPE': []
    }

    for result in results:
        for key, value in result.items():
            data[key].extend(value)

    # Clean up
    pool.close()
    pool.join()

    return data
