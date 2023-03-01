import config as conf

import string
import warnings
import shelve
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
    timepoint_index: int = 0,
    num_counties: int = 5,
    suppress_fit_warnings: bool = False
):

    # Load data column index
    index = shelve.open(paths.PARSED_DATA_COLUMN_INDEX)

    # Empty dict to hold results
    fitted_models = {
        'fitted_model': [],
        'AIC': [],
        'BIC': [],
        'lag_order': [],
        'difference_degree': [],
        'moving_average_order': [],
        'data_type_string': [],
        'block_size': [],
        'timepoint_index': [],
        'num_counties': []
    }

    # Loop on block sizes
    for block_size in block_sizes:

        # Load data with block size
        input_file = f'{paths.PARSED_DATA_PATH}/{params.input_file_root_name}{block_size}.npy'
        timepoints = np.load(input_file)

        # Loop on data types
        for data_type_string, data_type in zip(data_type_strings, data_types):

            # Loop on model parameter sets
            for model_parameter_set in product(lag_orders, difference_degrees, moving_average_orders):

                # Loop on example county indexes
                for i in range(num_counties):

                    with warnings.catch_warnings():

                        if suppress_fit_warnings == True:
                            warnings.simplefilter("ignore")

                        # Fit ARIMA model to data type for this county with model parameters
                        model = ARIMA(pd.Series(timepoints[timepoint_index,i,:,index[data_type]]), order=model_parameter_set)
                        model_fit = model.fit()
                        
                        # Add fitted model to results dict
                        fitted_models[f'fitted_model'].append(model_fit)

                        # Add AIC and BIC to results dict
                        fitted_models['AIC'].append(model_fit.aic)
                        fitted_models['BIC'].append(model_fit.bic)

                        # Add model and data details to results dict
                        fitted_models['lag_order'].append(model_parameter_set[0])
                        fitted_models['difference_degree'].append(model_parameter_set[1])
                        fitted_models['moving_average_order'].append(model_parameter_set[2])
                        fitted_models['data_type_string'].append(data_type_string)
                        fitted_models['block_size'].append(block_size)
                        fitted_models['timepoint_index'].append(timepoint_index)
                        fitted_models['num_counties'].append(num_counties)

    fitted_models_df = pd.DataFrame(fitted_models)

    return fitted_models_df

def plot_timeseries(
    data_types: list[str] = ['microbusiness_density'],
    data_type_strings: list[str] = ['raw MBD'],
    num_counties: int = 5,
    block_size: int = 37,
    plot_height: float = 2.25,
    fig_width: int = 8
):
    # Load data column index
    index = shelve.open(paths.PARSED_DATA_COLUMN_INDEX)

    # Load data with block size
    input_file = f'{paths.PARSED_DATA_PATH}/{params.input_file_root_name}{block_size}.npy'
    timepoints = np.load(input_file)

    fig, ax = plt.subplots(
        nrows = len(data_types), 
        ncols = 1, 
        figsize=(
            fig_width, # width
            len(data_types) * plot_height # height
        )
    )

    for plot_num, data_type in enumerate(zip(data_type_strings, data_types)):
        for i in range(num_counties):
                
            ax[plot_num].plot(list(range(len(timepoints[0,1,:,2]))), timepoints[0,i,:,index[data_type[1]]])

        ax[plot_num].set_xlabel(f'Timepoint')
        ax[plot_num].set_ylabel(data_type[0])

    plt.suptitle('Example time courses by county')
    plt.tight_layout()

    return plt

def plot_autocorrelation(
    data_types: list[str] = ['microbusiness_density'],
    data_type_strings: list[str] = ['raw MBD'],
    num_counties: int = 5,
    block_size: int = 37,
    plot_height: float = 2.25,
    fig_width: int = 8
):
    # Load data column index
    index = shelve.open(paths.PARSED_DATA_COLUMN_INDEX)

    # Load data with block size
    input_file = f'{paths.PARSED_DATA_PATH}/{params.input_file_root_name}{block_size}.npy'
    timepoints = np.load(input_file)

    fig, ax = plt.subplots(
        nrows = len(data_types), 
        ncols = 1, 
        figsize=(
            fig_width, # width
            len(data_types) * plot_height # height
        )
    )

    for plot_num, data_type in enumerate(zip(data_type_strings, data_types)):
        for i in range(num_counties):
                
            autocorrelation_plot(pd.Series(timepoints[0,i,:,index[data_type[1]]]), ax=ax[plot_num])

        ax[plot_num].set_title(data_type[0])

    plt.suptitle('Example autocorrelation by county')
    plt.tight_layout()

    return plt

def plot_residuals(
    fitted_models_df: pd.DataFrame,
    plot_rows: list[int] = [0],
    plot_cols: list[int] = [0],
    title_template: str = 'Model fit residuals',
    plot_dim: int = 2,
    plot_density: bool = False      
):
    grouped_fitted_models_df = fitted_models_df.groupby([
        'block_size',
        'num_counties',
        'data_type_string',
        'lag_order',
        'difference_degree',
        'moving_average_order'
    ])


    fig, ax = plt.subplots(
        nrows = len(plot_rows), 
        ncols = len(plot_cols), 
        figsize=(
        len(plot_cols) * plot_dim, # width
        len(plot_rows) * plot_dim  # height
        )
    )

    # loop on groupby object and product of row & column lists
    for model_group, subplot in zip(grouped_fitted_models_df, product(plot_rows, plot_cols)):

        # Get group name (containing model parameters) and dataframe from groupby
        model_parameters = model_group[0]
        model_df = model_group[1]

        # Extract model parameters from this group
        values = {
            'block_size': model_parameters[0],
            'num_counties': model_parameters[1],
            'data_type_string': model_parameters[2],
            'lag_order': model_parameters[3],
            'difference_degree': model_parameters[4],
            'moving_average_order': model_parameters[5]
        }

        for fitted_model in model_df['fitted_model']:
            residuals = pd.DataFrame(fitted_model.resid)

            if plot_density == False:
                residuals.plot(legend=False, ax=ax[subplot])
                ax[subplot].set_xlabel(f'Timepoint')
                ax[subplot].set_ylabel(f'Fit residual')

            elif plot_density == True:
                residuals.plot(kind='kde', legend=False, ax=ax[subplot])
                ax[subplot].set_xlabel(f'Fit residual')
                ax[subplot].set_ylabel(f'Density')

            t = string.Template(title_template)
            ax[subplot].set_title(t.substitute(values))

    if plot_density == False:
        plt.suptitle('ARIMA models: fit residuals timeseries', weight='bold')

    elif plot_density == True:
        plt.suptitle('ARIMA models: fit residuals density', weight='bold')

    plt.tight_layout()

    return plt

def model_performance_boxplot(
    fitted_models_df: pd.DataFrame,
    x_variable: str = 'data_type',
    x_axis_label: str = 'Model',
    hue_by: str = 'lag_order',
    fig_height: int = 6,
    fig_width: int = 12,
    rotate_x_axis_labels = False

):

    # Make the plots
    fig, ax = plt.subplots(
        nrows = 2,
        ncols = 1,
        figsize=(
            fig_width, # width
            fig_height # height
        )
    )

    sns.boxplot(
        data=fitted_models_df, 
        x=x_variable,
        y='AIC',
        hue=hue_by,
        ax=ax[0]
    )

    ax[0].set(
        xlabel=x_axis_label, 
        ylabel='AIC score', 
        title='Model AIC scores'
    )

    if rotate_x_axis_labels == True:
        ax[0].tick_params(axis='x', rotation=15)

    sns.boxplot(
        data=fitted_models_df, 
        x=x_variable,
        y='BIC',
        hue=hue_by,
        ax=ax[1]
    )

    ax[1].set(
        xlabel=x_axis_label, 
        ylabel='BIC score', 
        title='Model BIC scores'
    )

    if rotate_x_axis_labels == True:
        ax[1].tick_params(axis='x', rotation=15)

    plt.tight_layout()
    
    return plt

def parse_results(
    input_file: str = f'{paths.BOOTSTRAPPING_RESULTS_PATH}/{params.output_file_root_name}.parquet'      
):
    # Load data
    data_df = pd.read_parquet(input_file)

    # Calculate the final SMAPE score for each sample in each condition
    sample_smape_scores_df = data_df.groupby([
        'sample',
        'model_type',
        'block_size',
        'lag_order',
        'difference_degree',
        'moving_average_order'
    ])[['SMAPE_value']].mean().mul(100)

    # Rename columns to reflect the change from SMAPE values for a single prediction to
    # SMAPE scores within a sample
    sample_smape_scores_df.rename(inplace=True, columns={'SMAPE_value': 'SMAPE_score'})

    # Add log SMAPE score column for plotting
    sample_smape_scores_df['log_SMAPE_score'] = np.log(sample_smape_scores_df['SMAPE_score'].astype(float))

    # Clean up index and inspect. Now each sample in all of the conditions is represented by a single row
    # with two SMAPE scores calculated from all of the datapoints in that condition and sample. One from
    # the fit and forecast on the raw data and the other from the fit and forecast on the difference
    # detrended data
    sample_smape_scores_df.reset_index(inplace=True, drop=False)
    sample_smape_scores_df.head()

    return sample_smape_scores_df

def sample_mean_smape_boxplot(
    sample_smape_scores_df: pd.DataFrame,
    parameter: str = 'block_size',
    hue_by: str = 'difference_degree'
):

    # Get the best control mean of sample SMAPE scores from this experiment so that we can add
    # it to the plots for comparison
    mean_smape_score_df = sample_smape_scores_df.groupby(['model_type', 'lag_order', 'difference_degree', 'moving_average_order'])[['SMAPE_score']].mean()
    mean_smape_score_df.reset_index(inplace=True, drop=False)

    winning_control_mean_smape = mean_smape_score_df[mean_smape_score_df['model_type'] == 'control'].sort_values(by=['SMAPE_score'])
    winning_control_mean_smape.reset_index(inplace=True, drop=True)
    winning_control_mean_smape = winning_control_mean_smape['SMAPE_score'].iloc[0]

    # Also get the best single sample SMAPE score
    winning_control_smape = sample_smape_scores_df[sample_smape_scores_df['model_type'] == 'control'].sort_values(by=['SMAPE_score'])
    winning_control_smape.reset_index(inplace=True, drop=True)
    winning_control_smape = winning_control_smape['SMAPE_score'].iloc[0]

    print()
    print(f'Winning control mean of sample SMAPE scores: {winning_control_mean_smape}')
    print(f'Winning control single sample SMAPE score: {winning_control_smape}')
    print()

    # Get condition levels 
    parameters = {
        'lag_order': list(sample_smape_scores_df.lag_order.unique()),
        'difference_degree': list(sample_smape_scores_df.difference_degree.unique()),
        'moving_average_order': list(sample_smape_scores_df.moving_average_order.unique()),
        'block_size': list(sample_smape_scores_df.block_size.unique())
    }

    for parameter_name, levels in parameters.items():
        print(f'{parameter_name}: {levels}')

    print()

    # Find the experiment-wide max and min SMAPE score so that we can set a common Y-scale for all
    # of the subplots
    max_smape_score = sample_smape_scores_df[sample_smape_scores_df['model_type'] == 'ARIMA'].sort_values(by=['SMAPE_score'], ascending=False)
    max_smape_score.reset_index(inplace=True, drop=True)
    max_smape_score = max_smape_score['SMAPE_score'].iloc[0]

    min_smape_score = sample_smape_scores_df[sample_smape_scores_df['model_type'] == 'ARIMA'].sort_values(by=['SMAPE_score'], ascending=True)
    min_smape_score.reset_index(inplace=True, drop=True)
    min_smape_score = min_smape_score['SMAPE_score'].iloc[0]

    # Plot
    fig, ax = plt.subplots(len(parameters[parameter]), 1, figsize=(12, (3.5 * len(parameters[parameter]))))

    for plot_num, parameter_level in enumerate(parameters[parameter]):

        sns.boxplot(
            data=sample_smape_scores_df[(sample_smape_scores_df['model_type'] == 'ARIMA') & (sample_smape_scores_df[parameter] == parameter_level)], 
            x='lag_order',
            y='SMAPE_score',
            hue=hue_by,
            ax=ax[plot_num]
        )

        # Set common y limit
        ax[plot_num].set_ylim([min_smape_score - (0.05*min_smape_score), max_smape_score + (0.05*max_smape_score)])

        # Add line for best control sample mean
        ax[plot_num].axhline(winning_control_mean_smape, 0, 1, color='red', label=f'Best control mean of\nsample SMAPE scores')
        ax[plot_num].axhline(min_smape_score, 0, 1, color='red', linestyle='dashed',label=f'Best control single\nsample SMAPE score')
        
        ax[plot_num].set(yscale='log')

        # Labels
        ax[plot_num].set(
            xlabel='Lag order', 
            ylabel='sample SMAPE score', 
            title=f'{parameter} {parameter_level}'
        )

        # Put a legend to the right of the current axis
        ax[plot_num].legend(title=hue_by, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.tight_layout()
    
    return plt