import time
import logging
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import Ridge

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

def make_forecasts(block, model_types, model_order, time_fit = False):
    '''Uses specified model type and model order to forecast
    within block, one timepoint into the future. Also returns
    naive, 'carry-forward' prediction for the same datapoint 
    for comparison'''

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
    block_predictions['MBD_predictions'].append(block[(model_order - 1), 2])
    block_predictions['detrended_MBD_predictions'].append(block[(model_order - 1), 5] + block[model_order, 2])

    # X input is model_order sequential integers
    x_input = list(range(model_order))

    # Y input is MBD values starting from the left
    # edge of the block, up to the model order. Note: this
    # is a slice so, the right edge is exclusive 
    y_input = list(block[:model_order, 2])
    detrended_y_input = list(block[:model_order, 5])

    block_predictions['MBD_inputs'].append(y_input)
    block_predictions['detrended_MBD_inputs'].append(detrended_y_input)

    # Forecast X input is sequential integers starting
    # after the end of the X input. Note: we are only interested
    # in the first prediction here, but some statsmodels estimators
    # expect the same dim during forecast as they were fitted 
    forecast_x = list(range(model_order, (model_order * 2)))

    for model_type in model_types:

        start_time = time.time()

        if model_type == 'OLS':

            # Add model type to results
            block_predictions['model_type'].append(model_type)

            # Add model order to results
            block_predictions['model_order'].append(model_order)

            block_predictions['MBD_inputs'].append(y_input)
            block_predictions['detrended_MBD_inputs'].append(detrended_y_input)

            # Fit and predict raw data
            model = sm.OLS(y_input, sm.add_constant(x_input)).fit()
            prediction = model.predict(sm.add_constant(forecast_x))

            # Collect forecast
            block_predictions['MBD_predictions'].append(prediction[0])

            # Fit and predict detrended data
            model = sm.OLS(detrended_y_input, sm.add_constant(x_input)).fit()
            prediction = model.predict(sm.add_constant(forecast_x))

            # Collect forecast
            block_predictions['detrended_MBD_predictions'].append(prediction[0] + block[model_order, 2])

        if model_type == 'TS':

            # Add model type to results
            block_predictions['model_type'].append(model_type)

            # Add model order to results
            block_predictions['model_order'].append(model_order)

            block_predictions['MBD_inputs'].append(y_input)
            block_predictions['detrended_MBD_inputs'].append(detrended_y_input)

            # Fit Theil-Sen to raw data
            ts = stats.theilslopes(y_input, x_input)

            # Calculate forecast from Theil-Sen slope and intercept, add to results
            block_predictions['MBD_predictions'].append(ts[1] + ts[0] * forecast_x[0])

            # Fit Theil-Sen to detrended data
            ts = stats.theilslopes(detrended_y_input, x_input)

            # Calculate forecast from Theil-Sen slope and intercept, add to results
            block_predictions['detrended_MBD_predictions'].append((ts[1] + ts[0] * forecast_x[0]) + block[model_order, 2])

        if model_type == 'Seigel':

            # Add model type to results
            block_predictions['model_type'].append(model_type)

            # Add model order to results
            block_predictions['model_order'].append(model_order)

            block_predictions['MBD_inputs'].append(y_input)
            block_predictions['detrended_MBD_inputs'].append(detrended_y_input)

            # Fit Seigel to raw data
            ss = stats.siegelslopes(y_input, x_input)

            # Calculate forecast from Seigel slope and intercept, add to results
            block_predictions['MBD_predictions'].append(ss[1] + ss[0] * forecast_x[0])

            # Fit Theil-Sen to detrended data
            ss = stats.siegelslopes(detrended_y_input, x_input)

            # Calculate forecast from Seigel slope and intercept, add to results
            block_predictions['detrended_MBD_predictions'].append((ss[1] + ss[0] * forecast_x[0]) + block[model_order, 2])

        if model_type == 'Ridge':

            # Add model type to results
            block_predictions['model_type'].append(model_type)

            # Add model order to results
            block_predictions['model_order'].append(model_order)

            block_predictions['MBD_inputs'].append(y_input)
            block_predictions['detrended_MBD_inputs'].append(detrended_y_input)

            # Fit ridge to raw data
            ridge = Ridge()
            ridge.fit(np.array(x_input).reshape(-1, 1), np.array(y_input).reshape(-1, 1))

            # Get prediction, add to results
            block_predictions['MBD_predictions'].append(ridge.predict(np.array(forecast_x).reshape(-1, 1))[0][0])

            # Fit ridge to detrended data
            ridge = Ridge()
            ridge.fit(np.array(x_input).reshape(-1, 1), np.array(detrended_y_input).reshape(-1, 1))

            # Get prediction, add to results
            block_predictions['detrended_MBD_predictions'].append(ridge.predict(np.array(forecast_x).reshape(-1, 1))[0][0] + block[model_order, 2])

        dT = time.time() - start_time

        if time_fit == True:
            print(f'{model_type}, order {model_order}: {dT} sec.')  

    return block_predictions

def smape_score_models(sample, model_types, model_order, time_fit = False):
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
        block_predictions = make_forecasts(sample[block_num], model_types, model_order, time_fit)

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

def bootstrap_smape_scores(timepoints, sample_num, sample_size, model_order, model_types, time_fit = False):

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
    result = smape_score_models(sample, model_types, model_order, time_fit)

    # Add sample results
    for key, value in result.items():
        sample_data[key].extend(value)

    # Fill sample number in 
    sample_data['sample'].extend([sample_num] * len(result['model_type']))

    return sample_data