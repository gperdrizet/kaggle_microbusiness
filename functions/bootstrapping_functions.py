import time
import logging
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm

from scipy import stats
from sklearn.linear_model import Ridge
from statsmodels.tsa.arima.model import ARIMA

def two_point_smape(actual, forecast):
    '''Takes two datapoints and returns the SMAPE value for the pair'''

    # If SMAPE denominator is zero set SMAPE to zero
    if actual == 0 and forecast == 0:
        return 0

    # Calculate smape for forecast
    smape = abs(forecast - actual) / ((abs(actual) + abs(forecast)) / 2)
    
    return smape

def sample_parsed_data(timepoints, sample_size):
    '''Generates a random sample of sample_size from a random timepoint'''

    # Initialize random seed to make sure that output is differently random each call
    np.random.seed()

    # Pick random timepoint
    random_timepoint_index = np.random.choice(timepoints.shape[0], 1)
    timepoint = timepoints[random_timepoint_index][0]

    if sample_size == 'all':
        return timepoint

    # Pick n unique random county indexes to include in the sample
    random_county_indices = np.random.choice(timepoint.shape[0], sample_size, replace=False)

    # Use random indices to extract sample from timepoint
    sample = timepoint[random_county_indices]

    return sample

def make_forecasts(block, model_types, model_order, time_fits = False):
    '''Uses specified model type and model order to forecast
    within block, one timepoint into the future. Also returns
    naive, 'carry-forward' prediction for the same datapoint 
    for comparison'''

    # Log input block for debug
    logging.debug('')
    logging.debug(f'Input block:')

    for row in block[0:,]:
        row = [f'{x:.3e}' for x in row]
        logging.debug(f'{row}')

    logging.debug('')

    # Holder for SMAPE values
    block_predictions = {
        'model_type': [],
        'model_order': [],
        'MBD_predictions': [],
        'detrended_MBD_predictions': [],
        'MBD_inputs': [],
        'detrended_MBD_inputs': []
    }

    # Get prediction for naive control. Note: these are indexes
    # so model_order gets the model_order th element (zero anchored)
    block_predictions['model_type'].append('control')
    block_predictions['model_order'].append(model_order)

    control_prediction = block[(model_order - 1), 2]
    detrended_control_change_prediction = block[(model_order - 1), 5]
    detrended_control_prediction = detrended_control_change_prediction + block[(model_order - 1), 2]

    block_predictions['MBD_predictions'].append(control_prediction)
    block_predictions['detrended_MBD_predictions'].append(detrended_control_prediction)

    # X input is model_order sequential integers
    x_input = list(range(model_order))

    # Y input is MBD values starting from the left
    # edge of the block, up to the model order. Note: this
    # is a slice so, the right edge is exclusive 
    y_input = list(block[:model_order, 2])
    detrended_y_input = list(block[:model_order, 5])

    # Add input data to results
    block_predictions['MBD_inputs'].append(y_input)
    block_predictions['detrended_MBD_inputs'].append(detrended_y_input)

    # Forecast X input is sequential integers starting
    # after the end of the X input. Note: we are only interested
    # in the first prediction here, but some statsmodels estimators
    # expect the same dim during forecast as they were fitted 
    forecast_x = list(range(model_order, (model_order * 2)))

    true_y = block[model_order, 2]
    true_detrended_y = block[model_order, 5]

    # Log what we have so far legibly for debug
    formatted_y_input = [f'{x:.3f}' for x in y_input]
    formatted_detrended_y_input = [f'{x:.3f}' for x in detrended_y_input]

    logging.debug(f'MDB - input: {formatted_y_input}, target: {true_y:.3f}')
    logging.debug(f'Detrended MBD - input: {formatted_detrended_y_input}, target: {true_detrended_y:.3f}')
    logging.debug('')
    logging.debug(f'Control MBD prediction: {control_prediction:.3f}')
    logging.debug(f'Control detrended MBD prediction: {detrended_control_change_prediction:.3f}')
    logging.debug(f'Detrended control MBD prediction: {detrended_control_prediction:.3f}')

    for model_type in model_types:

        # Add model info. to results
        block_predictions['model_type'].append(model_type)
        block_predictions['model_order'].append(model_order)

        # Add input data to results
        block_predictions['MBD_inputs'].append(y_input)
        block_predictions['detrended_MBD_inputs'].append(detrended_y_input)

        # Set default values for outputs to NAN in case we have fits that fail
        # for some reason
        model_prediction = np.nan
        detrended_model_prediction = np.nan
        detrended_model_change_prediction = np.nan

        # Start fit timer
        start_time = time.time()

        if model_type == 'OLS':

            # Fit and predict raw data with OLS
            model = sm.OLS(y_input, sm.add_constant(x_input)).fit()
            model_predictions = model.predict(sm.add_constant(forecast_x))
            model_prediction = model_predictions[0]

            # Fit and predict detrended data with OLS
            model = sm.OLS(detrended_y_input, sm.add_constant(x_input)).fit()
            detrended_model_change_predictions = model.predict(sm.add_constant(forecast_x))
            detrended_model_change_prediction = detrended_model_change_predictions[0]
            detrended_model_prediction = detrended_model_change_prediction + block[(model_order - 1), 2]

        if model_type == 'TS':

            # Fit and predict raw data with Theil-Sen
            ts = stats.theilslopes(y_input, x_input)
            model_prediction = ts[1] + ts[0] * forecast_x[0]

            # Fit and predict detrended data with Theil-Sen
            ts = stats.theilslopes(detrended_y_input, x_input)
            detrended_model_change_prediction = ts[1] + ts[0] * forecast_x[0]
            detrended_model_prediction = detrended_model_change_prediction + block[(model_order - 1), 2]

        if model_type == 'Seigel':

            # Fit and predict raw data with Seigel
            ss = stats.siegelslopes(y_input, x_input)
            model_prediction = ss[1] + ss[0] * forecast_x[0]

            # Fit and predict detrended data with Seigel
            ss = stats.siegelslopes(detrended_y_input, x_input)
            detrended_model_change_prediction = ss[1] + ss[0] * forecast_x[0]
            detrended_model_prediction = detrended_model_change_prediction + block[(model_order - 1), 2]

        if model_type == 'Ridge':

            # Fit and predict raw data with ridge
            ridge = Ridge()
            ridge.fit(np.array(x_input).reshape(-1, 1), np.array(y_input).reshape(-1, 1))
            model_predictions = ridge.predict(np.array(forecast_x).reshape(-1, 1))[0]
            model_prediction = model_predictions[0]

            # Fit and predict detrended data with ridge
            ridge = Ridge()
            ridge.fit(np.array(x_input).reshape(-1, 1), np.array(detrended_y_input).reshape(-1, 1))
            detrended_model_change_predictions = ridge.predict(np.array(forecast_x).reshape(-1, 1))[0]
            detrended_model_change_prediction = detrended_model_change_predictions[0]
            detrended_model_prediction = detrended_model_change_prediction + block[(model_order - 1), 2]

        # Stop fit timer, get total dT in seconds
        dT = time.time() - start_time

        # Collect forecasts
        block_predictions['MBD_predictions'].append(model_prediction)
        block_predictions['detrended_MBD_predictions'].append(detrended_model_prediction)

        # Log model results for debug
        logging.debug('')
        logging.debug(f'{model_type} MBD prediction: {model_prediction:.3f}')
        logging.debug(f'{model_type} detrended MBD prediction: {detrended_model_change_prediction:.3f}')
        logging.debug(f'Detrended {model_type} MBD prediction: {detrended_model_prediction:.3f}')

        if time_fits == True:
            logging.info(f'{model_type}, order {model_order}: {dT:.2e} sec.') 

    return block_predictions

def make_ARIMA_forecasts(
    block,
    index,
    data_type,
    lag_order,
    difference_degree,
    moving_average_order,
    suppress_fit_warnings,
    time_fits
):
    '''Takes block, lag and moving average order and difference degree 
    Uses ARIMA to forecast from block, one timepoint into the future. 
    Also returns naive, 'carry-forward' prediction for the same datapoint 
    for comparison'''

    # Holder for SMAPE values
    block_predictions = {
        'model_type': [],
        'lag_order': [],
        'difference_degree': [],
        'moving_average_order': [],
        'MBD_prediction': [],
        'MBD_inputs': [],
        'fit_residuals': [],
        'AIC': [],
        'BIC': []
    }

    # Get input data from block, up to one datapoint before
    # the end of the timeseries (this will be the forecast)
    y_input = list(block[:-1, index[data_type]])

    # Make the 'control' prediction - i.e. the second to last
    # value from the block
    control_prediction = block[-1, index[data_type]]

    # Get prediction for naive control. Note: these are indexes
    # so model_order gets the model_order th element (zero anchored)
    block_predictions['model_type'].append('control')
    block_predictions['lag_order'].append(lag_order)
    block_predictions['difference_degree'].append(difference_degree)
    block_predictions['moving_average_order'].append(moving_average_order)
    block_predictions['MBD_inputs'].append(y_input)
    block_predictions['MBD_prediction'].append(control_prediction)

    # Add placeholder values for 'goodness-of-fit' columns for control
    block_predictions['fit_residuals'].append([0])
    block_predictions['AIC'].append(0)
    block_predictions['BIC'].append(0)

    # Add model info. to results
    block_predictions['model_type'].append('ARIMA')
    block_predictions['lag_order'].append(lag_order)
    block_predictions['difference_degree'].append(difference_degree)
    block_predictions['moving_average_order'].append(moving_average_order)
    block_predictions['MBD_inputs'].append(y_input)

    # Start fit timer
    start_time = time.time()

    with warnings.catch_warnings():

        if suppress_fit_warnings == True:
            warnings.simplefilter("ignore")

        model = ARIMA(y_input, order=(lag_order,difference_degree,moving_average_order))
        model_fit = model.fit()
        model_prediction = model_fit.forecast(steps = 1)[0]

    # Stop fit timer, get total dT in seconds
    dT = time.time() - start_time

    # Collect forecast
    block_predictions['MBD_prediction'].append(model_prediction)

    # Collect 'goodness-of-fit' results
    block_predictions['fit_residuals'].append(model_fit.resid)
    block_predictions['AIC'].append(model_fit.aic)
    block_predictions['BIC'].append(model_fit.bic)


    if time_fits == True:
        logging.info(f'ARIMA({lag_order}, {difference_degree}, {moving_average_order}): {dT:.2e} sec.') 

    return block_predictions

def smape_score_models(
        sample, 
        model_types, 
        model_order, 
        time_fits = False
):
    '''Takes a sample of blocks, makes forecast for each 
    and collects resulting SMAPE values'''

    # Holder for SMAPE values
    block_data = {
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

    for block_num in range(sample.shape[0]):

        # Get the forecasted value(s)
        block_predictions = make_forecasts(sample[block_num], model_types, model_order, time_fits)

        # Collect predictions, input data and model info.
        for key, value in block_predictions.items():
            block_data[key].extend(value)

        # Get the true value and add to data
        actual_value = sample[block_num, model_order, 2]

        # Get and collect SMAPE value for models
        for value in block_predictions['MBD_predictions']:

            smape_value = two_point_smape(actual_value, value)
            block_data['SMAPE_values'].append(smape_value)
            block_data['MBD_actual'].append(actual_value)

        # Get and collect SMAPE value for models
        for value in block_predictions['detrended_MBD_predictions']:

            smape_value = two_point_smape(actual_value, value)
            block_data['detrended_SMAPE_values'].append(smape_value)

    return block_data

def smape_score_ARIMA_models(
        sample,
        index,
        data_type,
        lag_order,
        difference_degree,
        moving_average_order,
        suppress_fit_warnings,
        time_fits
):
    '''Takes a sample of blocks, makes forecast for each 
    and collects resulting SMAPE values'''

    # Holder for SMAPE values
    block_data = {
        'model_type': [],
        'lag_order': [],
        'difference_degree': [],
        'moving_average_order': [],
        'SMAPE_value': [],
        'MBD_prediction': [],
        'MBD_inputs': [],
        'MBD_actual': [],
        'fit_residuals': [],
        'AIC': [],
        'BIC': []
    }

    for block_num in range(sample.shape[0]):

        # Get the forecasted value(s)
        block_predictions = make_ARIMA_forecasts(
            sample[block_num],
            index,
            data_type,
            lag_order,
            difference_degree,
            moving_average_order,
            suppress_fit_warnings,
            time_fits
        )

        # Collect predictions, input data and model info.
        for key, value in block_predictions.items():
            block_data[key].extend(value)

        # Get the true value and add to data
        actual_value = sample[block_num, lag_order, 2]

        # Get and collect SMAPE value for models
        for value in block_predictions['MBD_prediction']:

            smape_value = two_point_smape(actual_value, value)
            block_data['SMAPE_value'].append(smape_value)
            block_data['MBD_actual'].append(actual_value)

    return block_data

def bootstrap_smape_scores(
        timepoints, 
        sample_num, 
        sample_size, 
        model_order, 
        model_types, 
        time_fits = False
    ):
    '''Takes bootstrapping experiment run parameters, generates random sample of 
    blocks from timepoints and runs forecast/score for models+control, returns
    dict of results'''

    logging.debug('')
    logging.debug(f'Worker {sample_num} starting bootstrap run.')

    # Holder for sample results
    sample_data = {
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

    # Generate sample of random blocks from random timepoint
    sample = sample_parsed_data(timepoints, sample_size)

    # Do forecast and aggregate score across each block in sample
    result = smape_score_models(sample, model_types, model_order, time_fits)

    # Add sample results
    for key, value in result.items():
        sample_data[key].extend(value)

    # Fill sample number in 
    sample_data['sample'].extend([sample_num] * len(result['model_type']))

    return sample_data

def bootstrap_ARIMA_smape_scores(
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
    ):
    '''Takes bootstrapping experiment run parameters, generates random sample of 
    blocks from timepoints and runs forecast/score for models+control returns
    dict of results'''

    # logging.debug('')
    # logging.debug(f'Worker {sample_num} starting bootstrap run.')

    # Holder for sample results
    sample_data = {
        'sample': [],
        'model_type': [],
        'lag_order': [],
        'difference_degree': [],
        'moving_average_order': [],
        'SMAPE_value': [],
        'MBD_prediction': [],
        'MBD_inputs': [],
        'MBD_actual': [],
        'fit_residuals': [],
        'AIC': [],
        'BIC': []
    }

    # Generate sample of random blocks from random timepoint
    sample = sample_parsed_data(timepoints, sample_size)

    # Do forecast and aggregate score across each block in sample
    result = smape_score_ARIMA_models(
        sample,
        index,
        data_type,
        lag_order,
        difference_degree,
        moving_average_order,
        suppress_fit_warnings,
        time_fits
    )
    
    # Add sample results
    for key, value in result.items():
        sample_data[key].extend(value)

    # Fill sample number in using the number of lag orders reported in the
    # results dict as a proxy for the number of rows of results
    sample_data['sample'].extend([sample_num] * len(result['lag_order']))

    return sample_data