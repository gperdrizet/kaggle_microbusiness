import numpy as np
import pandas as pd
import statsmodels.api as sm
import multiprocessing as mp

from scipy import stats

def build_OLS_data(data, model_order):

    # Sort each county by timepoint and take the first n (most recent) rows from each county
    recent_values_df = data.sort_values('first_day_of_month', ascending=False).groupby('cfips').head(model_order)

    # Sort by cfips, then timepoint
    recent_values_df.sort_values(['cfips', 'first_day_of_month'], inplace=True)

    # Number timepoints so we have an easy numeric x variable for regression
    if 'timepoint_num' not in list(data.columns):
        recent_values_df.insert(1, 'timepoint_num', recent_values_df.groupby(['cfips']).cumcount())

    # Clean up
    recent_values_df.reset_index(inplace=True, drop=True)

    return recent_values_df


def OLS_prediction(data, xinput, yinput, xforecast):

    model = sm.OLS(data[yinput], sm.add_constant(data[xinput])).fit()
    predictions = model.predict(sm.add_constant(xforecast))

    return predictions

def siegel_prediction(data, xinput, yinput, xforecast):

    ss = stats.siegelslopes(data[yinput], data[xinput])

    predictions = []

    for x in xforecast:
        predictions.append(ss[1] + ss[0] * x)

    return predictions

def sample_parsed_data(
    parsed_data,
    training_fraction      
):
    '''Randomly select fraction of data for training, keep the rest for validation'''

    # Initialize random seed to make sure that output is differently random each call
    np.random.seed()

    # Calculate sample sizes
    training_sample_size = int(len(parsed_data) * training_fraction)

    # Generate list of random indices for training sample
    random_training_indices = np.random.choice(parsed_data.shape[0], training_sample_size, replace=False)

    # Use random indices to extract training sample from parsed data
    training_sample = parsed_data[random_training_indices]

    # Loop on parsed_data indices, if index was not in random validation sample
    # add that data to the validation sample
    validation_sample = []

    for i in range(len(parsed_data)):
        if i not in random_training_indices:
            validation_sample.append(parsed_data[i])

    # Convert to numpy array
    validation_sample = np.array(validation_sample)

    return training_sample, validation_sample

def two_point_smape(actual, forecast):

    # If SMAPE denominator is zero set SMAPE to zero
    if actual == 0 and forecast == 0:
        return 0

    # Calculate smape for forecast
    smape = abs(forecast - actual) / ((abs(actual) + abs(forecast)) / 2)
    
    return smape

def naive_model_smape_score(sample):

    # Holders SMAPE for forecast horizon of one and four
    one_point_smape_values = []
    four_point_smape_values = []

    for block in sample:

        # Get true forecast values
        forecast_value = block[0,-1,2]

        # Get the target values
        actual_values = block[1,0:,2]

        # Holder for this sample's SMAPE values
        smape_values = []

        # Calculate SMAPE value for forecast
        for actual_value in actual_values:
            smape_values.append(two_point_smape(actual_value, forecast_value))

        # Collect this sample's SMAPE values
        one_point_smape_values.append(smape_values[0])
        four_point_smape_values.extend(smape_values)

    # Calculate SMAPE score for this block
    one_point_smape_score = (100/len(one_point_smape_values)) * sum(one_point_smape_values)
    four_point_smape_score = (100/len(four_point_smape_values)) * sum(four_point_smape_values)

    return one_point_smape_score, four_point_smape_score

def naive_model_smape_score_2(sample, scored_timepoints):
    one_point_smape_values = []
    four_point_smape_values = []

    # Score only the first n timepoints from the sample
    for block in sample[:scored_timepoints]:

        # Get the last microbusiness_density value from the input block
        # and use this as our constant forecast
        forecast_value = block[0,-1,2]

        # Get the target values
        actual_values = block[1,0:,2]

        # Score the predictions
        smape_values = []

        for actual_value in actual_values:
            smape_values.append(two_point_smape(actual_value, forecast_value))

        one_point_smape_values.append(smape_values[0])
        four_point_smape_values.extend(smape_values)

    one_point_smape_score = (100/len(one_point_smape_values)) * sum(one_point_smape_values)
    four_point_smape_score = (100/len(four_point_smape_values)) * sum(four_point_smape_values)

    return one_point_smape_score, four_point_smape_score

def crossvalidation_smape(parsed_data, folds, training_fraction):
    '''Generates random training-validation split, forecasts and calculates SMAPE score
    folds number of times, returns dict of SMAPE scores for forecast horizon of one and four'''

    # Holder for results
    smape_scores = {
        'one_point_training': [],
        'one_point_validation': [],
        'four_point_training': [],
        'four_point_validation': []
    }

    # Loop on folds
    for i in range(folds):

        # Get random training and validation samples
        training_sample, validation_sample = sample_parsed_data(
            parsed_data,
            training_fraction
        )

        # Forecast on and score training and validation subsets
        one_point_training_smape, four_point_training_smape = naive_model_smape_score(training_sample)
        one_point_validation_smape, four_point_validation_smape = naive_model_smape_score(validation_sample)

        # Collect scores
        smape_scores['one_point_training'].append(one_point_training_smape)
        smape_scores['one_point_validation'].append(one_point_validation_smape)
        smape_scores['four_point_training'].append(four_point_training_smape)
        smape_scores['four_point_validation'].append(four_point_validation_smape)

    # Convert to pandas dataframe
    smape_scores_df = pd.DataFrame(smape_scores)

    return smape_scores_df

def parallel_crossvalidation_smape(parsed_data, training_fraction):
    '''Parallelized on CPUs over folds
    Generates random training-validation split, forecasts and calculates SMAPE score
    folds number of times, returns dict of SMAPE scores for forecast horizon of one and four'''

    # Get random training and validation samples
    training_sample, validation_sample = sample_parsed_data(
        parsed_data,
        training_fraction
    )

    # Forecast on and score training and validation subsets
    one_point_training_smape, four_point_training_smape = naive_model_smape_score(training_sample)
    one_point_validation_smape, four_point_validation_smape = naive_model_smape_score(validation_sample)

    # Collect scores
    result = [
        one_point_training_smape,
        one_point_validation_smape,
        four_point_training_smape,
        four_point_validation_smape
    ]

    return result

def parallel_crossvalidation_smape_2(parsed_data, training_fraction, scored_timepoints):
    '''Parallelized on CPUs over folds
    Generates random training-validation split, forecasts and calculates SMAPE score
    folds number of times, returns dict of SMAPE scores for forecast horizon of one and four'''

    # Get random training and validation samples
    training_sample, validation_sample = sample_parsed_data(
        parsed_data,
        training_fraction
    )

    # Forecast on and score training and validation subsets
    one_point_training_smape, four_point_training_smape = naive_model_smape_score_2(training_sample, scored_timepoints)
    one_point_validation_smape, four_point_validation_smape = naive_model_smape_score_2(validation_sample, scored_timepoints)

    # Collect scores
    result = [
        one_point_training_smape,
        one_point_validation_smape,
        four_point_training_smape,
        four_point_validation_smape
    ]

    return result

def start_multiprocessing_pool():
    # Instantiate multiprocessing pool to parallelize over folds
    n_cpus = mp.cpu_count() - 2

    print(f'Starting processes for {n_cpus} CPUs (available - 2)')

    pool = mp.Pool(processes = n_cpus)

    # Holder for result objects
    result_objects = []

    return pool, result_objects

def cleanup_multiprocessing_pool(pool, result_objects):

    # Collect results
    results = [result.get() for result in result_objects]

    # Convert to nice pandas dataframe
    smape_scores_df = pd.DataFrame(results, columns=[
        'one_point_training', 
        'one_point_validation', 
        'four_point_training', 
        'four_point_validation'
    ])

    # Clean up
    pool.close()
    pool.join()

    return smape_scores_df

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