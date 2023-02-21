import config as conf
import functions.initialization_functions as init_funcs
import functions.parallelization_functions as parallel_funcs

import logging
import pandas as pd

if __name__ == '__main__':

    # Instantiate paths and params
    paths = conf.DataFilePaths()
    params = conf.ARIMA_model_parameters()

    # Fire up logger
    logger = init_funcs.start_logger(
        logfile = f'{paths.LOG_DIR}/{params.log_file_name}',
        logstart_msg = 'Starting bootstrapped ARIMA optimization run'
    )

    # Log some details about the run
    logging.info('')
    logging.info(f'CPUs: {params.n_cpus}')
    logging.info(f'Samples: {params.num_samples} ({params.samples_per_cpu} per CPU)')
    logging.info(f'Sample size: {params.sample_size}')
    logging.info(f'Block sizes: {params.block_sizes}')
    logging.info(f'Lag orders: {params.lag_orders}')
    logging.info(f'Difference degrees: {params.difference_degrees}')
    logging.info(f'Moving average orders: {params.moving_average_orders}')
                 

    # Fire up the pool
    pool, result_objects = parallel_funcs.start_multiprocessing_pool()

    # Loop on samples, assigning each to a different worker
    for sample_num in range(params.num_samples):

        result = pool.apply_async(parallel_funcs.parallel_ARIMA_gridsearch,
            args = (
                paths.PARSED_DATA_PATH,
                params.input_file_root_name,
                sample_num, 
                params.sample_size,
                params.block_sizes,
                params.lag_orders,
                params.difference_degrees,
                params.moving_average_orders, 
                params.time_fits
            )
        )

        # Add result to collection
        result_objects.append(result)

    # Get and parse result objects, clean up pool
    data = parallel_funcs.cleanup_ARIMA_bootstrapping_multiprocessing_pool(pool, result_objects)

    for key, value in data.items():
        print(f'{key}: {len(value)}')

    # Convert result to Pandas DataFrame
    data_df = pd.DataFrame(data)

    # Persist to disk as HDF5
    output_file = f'{paths.BOOTSTRAPPING_RESULTS_PATH}/{params.output_file_root_name}.parquet'
    data_df.to_parquet(output_file)