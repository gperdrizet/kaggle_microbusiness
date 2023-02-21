import config as conf
import functions.initialization_functions as init_funcs
import functions.parallelization_functions as parallel_funcs

import pandas as pd

if __name__ == '__main__':

    paths = conf.DataFilePaths()
    params = conf.ARIMA_model_parameters()

    logger = init_funcs.start_logger(
        logfile = f'{paths.LOG_DIR}/{params.log_file_name}',
        logstart_msg = 'Starting bootstrapped ARIMA optimization run'
    )

    # Block size used for parsed data loading needs to be 
    # the largest model model order plus one for the forecast
    block_size = max(params.lag_orders + params.moving_average_orders) + 1

    # Now load up the parsed data with the blocksize calculated above and log some run details
    timepoints = init_funcs.load_inspect_parsed_data(
        input_file = f'{paths.PARSED_DATA_PATH}/{params.input_file_root_name}{block_size}.npy',
        params = params
    )

    # Fire up the pool
    pool, result_objects = parallel_funcs.start_multiprocessing_pool()

    # Loop on samples, assigning each to a different worker
    for sample_num in range(params.num_samples):

        result = pool.apply_async(parallel_funcs.parallel_ARIMA_gridsearch,
            args = (
                timepoints, 
                sample_num, 
                params.sample_size, 
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

    # Convert result to Pandas DataFrame
    data_df = pd.DataFrame(data)

    # Persist to disk as HDF5
    output_file = f'{paths.BOOTSTRAPPING_RESULTS_PATH}/{params.output_file_root_name}.parquet'
    data_df.to_parquet(output_file)