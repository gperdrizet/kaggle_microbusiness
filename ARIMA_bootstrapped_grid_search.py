import config as conf
import functions.initialization_functions as init_funcs
#import functions.parallelization_functions as parallel_funcs

if __name__ == '__main__':

    paths = conf.DataFilePaths()
    params = conf.ARIMA_model_parameters()

    logger = init_funcs.start_logger(
        logfile = f'{paths.LOG_DIR}/{params.log_file_name}',
        logstart_msg = 'Starting bootstrapped ARIMA optimiztion run'
    )

    # Block size used for parsed data loading needs to be 
    # the largest model model order plus one for the forecast
    block_size = max(params.model_orders) + 1

    timepoints = init_funcs.load_inspect_parsed_data(
        input_file = f'{paths.PARSED_DATA_PATH}/{params.input_file_root_name}{block_size}.npy',
        params = params
    )