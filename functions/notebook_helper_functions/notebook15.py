# import config as conf

import logging
import numpy as np
# import string
# import warnings
# import shelve
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers

# from itertools import product
# from pandas.plotting import autocorrelation_plot
# from statsmodels.tsa.arima.model import ARIMA

# paths = conf.DataFilePaths()
# params = conf.GRU_model_parameters()


def training_validation_testing_split(
    index,
    timepoints,
    num_counties: str = 'all',
    input_data_type: str = 'microbusiness_density',
    testing_timepoints: int = 1,
    training_split_fraction: float = 0.7
):
    '''Does training, validation, testing split on data. Also subsets feature
    columns and counties. Splits on time axis so that training data is time distal,
    validation is time proximal and test time points are the most recent.
    Returns dataset as dict of numpy with 'training', 'validation' and 'testing' 
    as keys.'''

    logging.info('')
    logging.info('####### TRAINING, VALIDATION, TESTING SPLIT ################')
    
    # Empty dict for results
    datasets = {}

    # Before we split, choose just the data we want and drop everything else
    if num_counties == 'all':
        input_data = timepoints[:,:,:,[index[input_data_type]]]
    else:
        input_data = timepoints[:,:num_counties,:,[index[input_data_type]]]

    # Reserve last n timepoints for true hold-out test set if needed
    if testing_timepoints != None:
        testing_data = input_data[-testing_timepoints:,:,:,:]
        input_data = input_data[:-testing_timepoints,:,:,:]

        datasets['testing'] = testing_data

    # Choose split
    split_index = int(input_data.shape[0] * training_split_fraction)

    # Split data into training and validation sets using chosen index. First 
    # portion becomes training, second portion is validation
    training_data = input_data[0:split_index]
    validation_data = input_data[split_index:-1]

    datasets['training'] = training_data
    datasets['validation'] = validation_data

    logging.info('')
    logging.info(f'Input data shape: {input_data.shape}')
    logging.info('')
    logging.info(f'Testing timepoints: {testing_timepoints}')
    logging.info(f'Split fraction: {training_split_fraction}')
    logging.info(f'Split index: {split_index}')
    logging.info(f'Training data shape: {training_data.shape}')
    logging.info(f'Validation data shape: {validation_data.shape}')

    if testing_timepoints != None:
        logging.info(f'Testing data shape: {testing_data.shape}')

    return datasets

def standardize_datasets(datasets):
    '''Uses mean and standard deviation from training data only to 
    convert training, validation and testing data to z-scores'''

    logging.info('')
    logging.info('####### DATA STANDARDIZATION ###############################')

    # Get mean and standard deviation from training data
    training_mean = np.mean(datasets['training'])
    training_deviation = np.std(datasets['training'])

    logging.info('')
    logging.info(f'Training data mean: {training_mean:.2f}, standard deviation: {training_deviation:.2f}')

    # Standardize the training, validation and test data
    datasets['training'] = (datasets['training'] - training_mean) / training_deviation
    datasets['validation'] = (datasets['validation'] - training_mean) / training_deviation
    datasets['testing'] = (datasets['testing'] - training_mean) / training_deviation

    logging.info('')
    logging.info(f"Training data, new mean: {np.mean(datasets['training']):.2f}, new standard deviation: {np.std(datasets['training']):.2f}")
    logging.info(f"Validation data, new mean: {np.mean(datasets['validation']):.2f}, new standard deviation: {np.std(datasets['validation']):.2f}")
    logging.info(f"Testing data, new mean: {np.mean(datasets['testing']):.2f}, new standard deviation: {np.std(datasets['testing']):.2f}")

    return datasets

def make_batch_major(datasets):
    '''Makes datasets batch major by swapping 0th and 2nd axis'''

    logging.info('')
    logging.info('####### SWAPPING BATCH AXIS ################################')
    logging.info('')

    for data_type, data in datasets.items():
        datasets[data_type] = np.swapaxes(data, 1, 0)
        logging.info(f'{data_type} data new shape: {datasets[data_type].shape}')

    return datasets

def build_GRU(
    GRU_units: int = 64,
    learning_rate: float = 0.0002,
    input_shape: list[int] = [13,8,1],
    output_units: int = 5
):
    '''Builds GRU based neural network for
    microbusiness density regression'''

    logging.info('')
    logging.info('####### BUILDING MODEL #####################################')
    logging.info('')

    # Input layer
    input = layers.Input(
        name = 'Input',
        shape = input_shape
    )

    # GRU layer
    gru = layers.GRU(
        GRU_units,
        activation='tanh',
        recurrent_activation='sigmoid',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        recurrent_initializer='orthogonal',
        bias_initializer='zeros',
        kernel_regularizer=None,
        recurrent_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        recurrent_constraint=None,
        bias_constraint=None,
        dropout=0.0,
        recurrent_dropout=0.0,
        return_sequences=False,
        return_state=False,
        go_backwards=False,
        stateful=False,
        unroll=False,
        time_major=False,
        reset_after=True,
        name='GRU'
    )(input)

    # output layer
    output = layers.Dense(
        name = 'Output',
        units = output_units,
        activation = 'linear'
    )(gru)

    # Next, we will build the complete model and compile it.
    model = keras.models.Model(
        input, 
        output,
        name = 'Simple_GRU_model'
    )

    model.compile(
        loss = keras.losses.MeanSquaredError(name = 'MSE'), 
        optimizer = keras.optimizers.Adam(learning_rate = learning_rate),
        metrics = [keras.metrics.MeanAbsoluteError(name = 'MAE')]
    )

    model.summary(print_fn=logging.info)

    return model

def data_generator(data, forecast_horizon):
    '''Generates pairs of X, Y (input, target) datapoints
    from batch major data'''
    
    # Don't stop
    while True:
        # Loop on counties
        for i in range(data.shape[0]):
                
            X = data[i,:,:-forecast_horizon,:]
            Y = data[i,:,-forecast_horizon:,:]

            yield (X, Y)

def train_GRU(
    datasets,
    forecast_horizon: int = 5,
    epochs: int = 10,
    GRU_units: int = 64,
    learning_rate: float = 0.002,

):
    '''Does single training run of GRU based neural network'''

    # Start generators and grab a sample data points to
    # derive network parameters
    training_data_generator = data_generator(datasets['training'], forecast_horizon)
    training_sample = next(training_data_generator)

    validation_data_generator = data_generator(datasets['validation'], forecast_horizon)
    validation_sample = next(validation_data_generator)

    # Check dataset's overall shape
    logging.info('')
    logging.info('####### DATA SHAPE ##########################################')
    logging.info('')
    logging.info(f"Training dataset: {datasets['training'].shape}")
    logging.info(f"Validation dataset: {datasets['validation'].shape}")
    logging.info('')
    logging.info(f'Training input: {training_sample[0].shape}')
    logging.info(f'Training target: {training_sample[1].shape}')
    logging.info('')
    logging.info(f'Validation input: {validation_sample[0].shape}')
    logging.info(f'Validation target: {validation_sample[1].shape}')

    # Check run parameters
    # Run parameters
    output_units = training_sample[1].shape[1]
    training_batch_size = datasets['training'].shape[1]
    training_batches = datasets['training'].shape[0]
    validation_batch_size = datasets['validation'].shape[1]
    validation_batches = datasets['validation'].shape[0]

    logging.info('')
    logging.info('####### RUN PARAMETERS #####################################')
    logging.info('')
    logging.info(f'Epochs: {epochs}')
    logging.info(f'Training batch size: {training_batch_size}')
    logging.info(f'Training batches: {training_batches}')
    logging.info(f'Validation batch size: {validation_batch_size}')
    logging.info(f'Validation batches: {validation_batches}')

    # Build the model
    model = build_GRU(
        GRU_units = GRU_units,
        learning_rate = learning_rate,
        input_shape = [training_sample[0].shape[1],training_sample[0].shape[2]],
        output_units = output_units
    )

    # Re-fire data generators for training and validation
    training_data_generator = data_generator(datasets['training'], forecast_horizon)
    validation_data_generator = data_generator(datasets['validation'], forecast_horizon)

    # Train the model
    history = model.fit(
        training_data_generator,
        epochs = epochs,
        batch_size = training_batch_size,
        steps_per_epoch = training_batches,
        validation_data = validation_data_generator,
        validation_batch_size = validation_batch_size,
        validation_steps = validation_batches
    )

    return model, history

def make_predictions(model, datasets, forecast_horizon):
    '''Uses a trained model to make predictions for training, 
    validation and testing datasets. Returns dict containing
    un-standardized and flattened predictions and targets for each'''

    logging.info('')
    logging.info('####### MAKING PREDICTIONS #################################')

    # Fire up generators
    training_data_generator = data_generator(datasets['training'], forecast_horizon)
    validation_data_generator = data_generator(datasets['validation'], forecast_horizon)
    testing_data_generator = data_generator(datasets['testing'], forecast_horizon)

    # Predict from training
    training_predictions = model.predict(
        training_data_generator,
        batch_size = datasets['training'].shape[1],
        steps = datasets['training'].shape[0]
    )

    # Predict from validation
    validation_predictions = model.predict(
        validation_data_generator,
        batch_size = datasets['validation'].shape[1],
        steps = datasets['validation'].shape[0]
    )

    # Predict from testing
    testing_predictions = model.predict(
        testing_data_generator,
        batch_size = datasets['testing'].shape[1],
        steps = datasets['testing'].shape[0]
    )

    # Get targets
    training_targets = datasets['training'][:,:,-forecast_horizon:,:]
    validation_targets = datasets['validation'][:,:,-forecast_horizon:,:]
    testing_targets = datasets['testing'][:,:,-forecast_horizon:,:]

    # Get mean and standard deviation from training data
    training_mean = np.mean(datasets['training'])
    training_deviation = np.std(datasets['training'])

    # Unstandardize everything
    training_predictions = (training_predictions * training_deviation) + training_mean
    training_targets = (training_targets * training_deviation) + training_mean
    validation_targets = (validation_targets * training_deviation) + training_mean
    validation_predictions = (validation_predictions * training_deviation) + training_mean
    testing_targets = (testing_targets * training_deviation) + training_mean
    testing_predictions = (testing_predictions * training_deviation) + training_mean

    logging.info('')
    logging.info(f"Training dataset: {datasets['training'].shape}")
    logging.info(f'Training targets: {training_targets.shape}')
    logging.info(f'training predictions: {training_predictions.shape}')
    logging.info('')
    logging.info(f"Validation dataset: {datasets['validation'].shape}")
    logging.info(f'Validation targets: {validation_targets.shape}')
    logging.info(f'Validation predictions: {validation_predictions.shape}')
    logging.info('')
    logging.info(f"Testing dataset: {datasets['testing'].shape}")
    logging.info(f'Testing targets: {testing_targets.shape}')
    logging.info(f'Testing predictions: {testing_predictions.shape}')

    # Flatten everything and add to predictions dict
    predictions = {}

    predictions['training_predictions'] = training_predictions
    predictions['training_targets'] = training_targets
    predictions['validation_predictions'] = validation_predictions
    predictions['validation_targets'] = validation_targets
    predictions['testing_predictions'] = testing_predictions
    predictions['testing_targets'] = testing_targets

    return predictions