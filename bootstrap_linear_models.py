import config as conf
import functions.initialization_functions as init_funcs
import functions.parallelization_functions as parallel_funcs

import logging
import numpy as np
import pandas as pd

if __name__ == '__main__':

    paths = conf.DataFilePaths()
    params = conf.LinearModelsBootstrappingParameters()

    logger = init_funcs.start_logger(
        logfile = f'{paths.LOG_DIR}/{params.log_file_name}',
        logstart_msg = 'Starting bootstrapping run'
    )

    # Block size used for parsed data loading needs to be 
    # the largest model model order plus one for the forecast
    block_size = max(params.model_orders) + 1

    # Load parsed data
    input_file = f'{paths.PARSED_DATA_PATH}/{params.input_file_root_name}{block_size}.npy'
    timepoints = np.load(input_file)

    # Log some details about the input data & run parameters
    logging.info('')
    logging.info(f'CPUs: {params.n_cpus}')
    logging.info(f'Samples: {params.num_samples} ({params.samples_per_cpu} per CPU)')
    logging.info(f'Sample size: {params.sample_size}')
    logging.info(f'Model orders: {params.model_orders}')
    logging.info(f'Model types: control + {params.model_types}')
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

    # Fire up the pool
    pool, result_objects = parallel_funcs.start_multiprocessing_pool()

    # Loop on samples, assigning each to a different worker
    for sample_num in range(params.num_samples):

        result = pool.apply_async(parallel_funcs.parallel_bootstrapped_smape,
            args = (
                timepoints, 
                sample_num, 
                params.sample_size, 
                params.model_orders, 
                params.model_types,
                params.time_fits
            )
        )

        # Add result to collection
        result_objects.append(result)

    # Get and parse result objects, clean up pool
    data = parallel_funcs.cleanup_bootstrapping_multiprocessing_pool(pool, result_objects)

    # Convert result to Pandas DataFrame
    data_df = pd.DataFrame(data)

    # Persist to disk as HDF5
    output_file = f'{paths.BOOTSTRAPPING_RESULTS_PATH}/{params.output_file_root_name}.parquet'
    data_df.to_parquet(output_file)