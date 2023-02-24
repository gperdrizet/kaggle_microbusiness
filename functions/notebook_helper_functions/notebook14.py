
import config as conf

import warnings
import shelve
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from itertools import product
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima.model import ARIMA

paths = conf.DataFilePaths()
params = conf.ARIMA_model_parameters()

def ARIMA_optimization(
    data_types: list[str] = ['microbusiness_density'],
    data_type_strings: list[str] = ['MBD'],
    lag_orders: list[int] = [2],
    difference_degrees: list[int] = [0],
    moving_average_orders: list[int] = [0],
    block_sizes: list[int] = [37],
    num_counties: int = 5,
    suppress_fit_warnings: bool = False
):

    # Load data column index
    index = shelve.open(paths.PARSED_DATA_COLUMN_INDEX)

    # Empty dict to hold results
    fitted_models = {}

    # Loop on block sizes
    for block_size in block_sizes:

        # Load data with block size
        input_file = f'{paths.PARSED_DATA_PATH}/{params.input_file_root_name}{block_size}.npy'
        timepoints = np.load(input_file)

        # Loop on data types
        for data_type_string, data_type in zip(data_type_strings, data_types):

            # Loop on model parameter sets
            for model_parameter_set in product(lag_orders, difference_degrees, moving_average_orders):

                # Add key for this set of test parameters to results dict
                fitted_models[f'{data_type_string} ARIMA{model_parameter_set}'] = []

                # Loop on example county indexes
                for i in range(num_counties):

                    with warnings.catch_warnings():

                        if suppress_fit_warnings == True:
                            warnings.simplefilter("ignore")

                        # Fit ARIMA model to data type for this county with model parameters
                        model = ARIMA(pd.Series(timepoints[0,i,:,index[data_type]]), order=model_parameter_set)
                        model_fit = model.fit()
                        
                        # Add fitted model to list in dict under key
                        fitted_models[f'{data_type_string} ARIMA{model_parameter_set}'].append(model_fit)

    return fitted_models

def plot_timeseries(
    data_types: list[str] = [None],
    data_type_strings: list[str] = [None],
    num_counties: int = 5,
    block_size: int = 37,
    plot_dim: int = 2,
    fig_width: int = 8
):
    # Load data column index
    index = shelve.open(paths.PARSED_DATA_COLUMN_INDEX)

    # Load data with block size
    input_file = f'{paths.PARSED_DATA_PATH}/{params.input_file_root_name}{block_size}.npy'
    timepoints = np.load(input_file)

    fig, ax = plt.subplots(len(data_types), 1, figsize=(fig_width,(len(data_types) * plot_dim)))

    for plot_num, data_type in enumerate(zip(data_type_strings, data_types)):
        for i in range(num_counties):
                
            ax[plot_num].plot(list(range(len(timepoints[0,1,:,2]))), timepoints[0,i,:,index[data_type[1]]])

        ax[plot_num].set_xlabel(f'Timepoint')
        ax[plot_num].set_ylabel(data_type[0])

    plt.suptitle('Example time courses by county')
    plt.tight_layout()

    return plt

def plot_autocorrelations(
    data_types: list[str] = [None],
    data_type_strings: list[str] = [None],
    num_counties: int = 5,
    block_size: int = 37,
    plot_dim: int = 2,
    fig_width: int = 8
):
    # Load data column index
    index = shelve.open(paths.PARSED_DATA_COLUMN_INDEX)

    # Load data with block size
    input_file = f'{paths.PARSED_DATA_PATH}/{params.input_file_root_name}{block_size}.npy'
    timepoints = np.load(input_file)

    fig, ax = plt.subplots(len(data_types), 1, figsize=(fig_width, (len(data_types) * plot_dim)))

    for plot_num, data_type in enumerate(zip(data_type_strings, data_types)):
        for i in range(num_counties):
                
            autocorrelation_plot(pd.Series(timepoints[0,i,:,index[data_type[1]]]), ax=ax[plot_num])

        ax[plot_num].set_title(data_type[0])

    plt.suptitle('Example autocorrelation by county')
    plt.tight_layout()

    return plt

def plot_residuals(
    fitted_models: dict = None,
    plot_rows: list[int] = [None],
    plot_cols: list[int] = [None],
    plot_dim: int = 2,
    num_counties: int = 5,
    plot_density: bool = False      
):

    fig, ax = plt.subplots(len(plot_rows), len(plot_cols), figsize=(len(plot_rows) * plot_dim, len(plot_rows) * plot_dim))

    for (model_string, fitted_model), subplot in zip(fitted_models.items(), product(plot_rows, plot_cols)):

        for i in range(num_counties):
            residuals = pd.DataFrame(fitted_models[model_string][i].resid)

            if plot_density == False:
                residuals.plot(legend=False, ax=ax[subplot])
                ax[subplot].set_xlabel(f'Timepoint')
                ax[subplot].set_ylabel(f'Fit residual')

            elif plot_density == True:
                residuals.plot(kind='kde', legend=False, ax=ax[subplot])
                ax[subplot].set_xlabel(f'Fit residual')
                ax[subplot].set_ylabel(f'Density')

            ax[subplot].set_title(model_string)

    if plot_density == False:
        plt.suptitle('ARIMA models: fit residuals timeseries')

    elif plot_density == True:
        plt.suptitle('ARIMA models: fit residuals density')

    plt.tight_layout()

    return plt