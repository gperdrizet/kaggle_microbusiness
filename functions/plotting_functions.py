import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.graphics.regressionplots import abline_plot

def two_panel_histogram(
        data: pd.Series,
        main_title: str,
        linear_plot_x_label: str,
        log10_plot_x_label: str,
        bins = 50,
        fig_x_dim = 8,
        fig_y_dim = 4
):

    fig, ax = plt.subplots(1, 2, figsize=(fig_x_dim, fig_y_dim))

    ax[0].hist(data, bins=bins)
    ax[0].set_xlabel(linear_plot_x_label)
    ax[0].set_ylabel('count')

    ax[1].hist(np.log10(data), bins=bins)
    ax[1].set_xlabel(log10_plot_x_label)
    ax[1].set_ylabel('count')

    plt.suptitle(main_title)
    plt.tight_layout()

    return plt

def timeseries_lag_two_panel_plot(
    timeseries_x_data: pd.Series,
    timeseries_y_data: pd.Series,
    main_title: str,
    timeseries_y_label: str,
    timeseries_x_label = 'timepoints',
    lag_x_label = 'y(t)',
    lag_y_label = 'y(t + 1)',
    fig_x_dim = 8,
    fig_y_dim = 4
):

    fig, ax = plt.subplots(1, 2, figsize=(fig_x_dim,fig_y_dim))

    ax[0].scatter(timeseries_x_data, timeseries_y_data)
    ax[0].set_xlabel(timeseries_x_label)
    ax[0].set_ylabel(timeseries_y_label)

    ax[1].scatter(timeseries_y_data, timeseries_y_data.shift(1))
    ax[1].set_xlabel(lag_x_label)
    ax[1].set_ylabel(lag_y_label)

    ax[0].set_xticks(ax[0].get_xticks())
    ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45)

    plt.suptitle(main_title)
    plt.tight_layout()

    return plt

def timeseries_percentage_two_panel_plot(
    timeseries_x_data: pd.Series,
    timeseries_y_data: pd.Series,
    main_title: str,
    timeseries_y_label: str,
    timeseries_x_label = 'timepoints',
    percentage_x_label = 'timepoints',
    percentage_y_label = 'percentage',
    fig_x_dim = 8,
    fig_y_dim = 4

):

    fig, ax = plt.subplots(1, 2, figsize=(fig_x_dim,fig_y_dim))

    ax[0].scatter(
        timeseries_x_data, 
        timeseries_y_data
    )

    ax[0].set_xlabel(timeseries_x_label)
    ax[0].set_ylabel(timeseries_y_label)


    ax[1].scatter(
        timeseries_x_data, 
        (timeseries_y_data / timeseries_y_data.sum()) * 100
    )

    ax[1].set_xlabel(percentage_x_label)
    ax[1].set_ylabel(percentage_y_label)

    ax[0].set_xticks(ax[0].get_xticks())
    ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45)

    ax[1].set_xticks(ax[0].get_xticks())
    ax[1].set_xticklabels(ax[0].get_xticklabels(), rotation=45)

    plt.suptitle(main_title)
    plt.tight_layout()
    
    return plt

def n_by_n_regression_plot(
    input_data: pd.DataFrame,
    x_variable: str = 'timepoint_num',
    y_variable: str = 'microbusiness_density',
    xlabel: str = 'Timepoint number',
    ylabel: str = 'Microbusiness density',
    cfips_list: list = [],
    rows: int = 1,
    columns: int = 1,
    main_title: str = 'Microbusiness density timeseries regression',
    set_const_ylims: bool = False,
    add_regression_line: bool = False,
    add_regression_stats: bool = False
):
    # Set common y-axis limits for all plots if desired
    if set_const_ylims == True:
        
        data_pool = []
        for cfips in cfips_list:
            data_pool.extend(input_data[input_data['cfips'] == cfips][y_variable].to_list())
        
        ymin = min(data_pool)
        ymax = max(data_pool)

    # Get plot dimensions based on number of rows and columns
    plot_width = columns * 3
    plot_height = rows * 3

    plot_num = 0

    fig, ax = plt.subplots(rows, columns, figsize=(plot_width,plot_height))

    for j in range(rows):
        for i in range(columns):
            if plot_num < len(cfips_list):

                data = input_data[input_data['cfips'] == cfips_list[plot_num]]

                x = data[x_variable]
                y = data[y_variable]

                ax[j,i].scatter(x, y) # type: ignore
                
                if add_regression_line == True:
                    result = sm.OLS(y, sm.add_constant(x)).fit()
                    abline_plot(model_results=result, ax=ax[j,i]) # type: ignore

                if add_regression_stats == True:
                    result = sm.OLS(y, sm.add_constant(x)).fit()
                    
                    coeff = result.params[1]
                    rsquared = result.rsquared
                    pvalue = result.pvalues[1]
                
                    ax[j,i].text( # type: ignore
                        0.05,
                        0.95, 
                        f'm: {coeff:.1e}\nR$^2$: {rsquared:.2f}\np: {pvalue:.1e}', 
                        horizontalalignment='left', 
                        verticalalignment='top', 
                        transform=ax[j,i].transAxes, # type: ignore
                        bbox = dict(facecolor = 'lightgrey', alpha = 1),
                        fontsize = 8
                    )

                if set_const_ylims == True:
                    ax[j,i].set_ylim(ymin, ymax) # type: ignore

                # If x axis is date, rotate tick labels and change font size
                if type(x.iloc[0]) == pd._libs.tslibs.timestamps.Timestamp: # type: ignore
                    ax[j,i].set_xticks(ax[j,i].get_xticks()) # type: ignore
                    ax[j,i].set_xticklabels(ax[j,i].get_xticklabels(), rotation=45, fontsize=8) # type: ignore

                ax[j,i].set_xlabel(xlabel) # type: ignore
                ax[j,i].set_ylabel(ylabel) # type: ignore

            plot_num += 1

    plt.suptitle(main_title)
    plt.tight_layout()

    return plt

def n_by_n_prediction_scatterplot(
    input_data: pd.DataFrame,
    predictions: pd.DataFrame,
    x_variable: str = 'timepoint_num',
    y_variable: str = 'microbusiness_density',
    xlabel: str = 'Timepoint number',
    ylabel: str = 'Microbusiness density',
    cfips_list: list = [],
    rows: int = 1,
    columns: int = 1,
    main_title: str = 'Microbusiness density timeseries',
    fig_width: int = 10,
    fig_height: int = 10,
    set_const_ylims: bool = False
):
    # Set common y-axis limits for all plots if desired
    if set_const_ylims == True:
        
        # Pool all of the y data we are going to plot
        data_pool = []

        for cfips in cfips_list:
            data_pool.extend(input_data[input_data['cfips'] == cfips][y_variable].to_list())
            data_pool.extend(predictions[predictions['cfips'] == cfips][y_variable].to_list())
        
        # Find min and max values from data pool
        ymin = min(data_pool)
        ymax = max(data_pool)

    # Count plots
    plot_num = 0

    # Initialize figure
    fig, ax = plt.subplots(rows, columns, figsize=(fig_width,fig_height))

    # Loop first on rows, then on columns
    for j in range(rows):
        for i in range(columns):

            # Plot if we have not run out of counties
            if plot_num < len(cfips_list):

                # Get input data for this county
                data = input_data[input_data['cfips'] == cfips_list[plot_num]]

                # Assign x and y from variable parameters
                x_input = data[x_variable]
                y_input = data[y_variable]

                # Plot
                ax[j,i].scatter(x_input, y_input) # type: ignore

                # Get prediction data for this county
                data = predictions[predictions['cfips'] == cfips_list[plot_num]]

                # Assign x and y from variable parameters
                x_predicted = data[x_variable]
                y_predicted = data[y_variable]

                # Plot
                ax[j,i].scatter(x_predicted, y_predicted) # type: ignore

                # Set axis labels
                ax[j,i].set_xlabel(xlabel) # type: ignore
                ax[j,i].set_ylabel(ylabel) # type: ignore

                # Set constant y axis limits, if desired
                if set_const_ylims == True:
                    ax[j,i].set_ylim(ymin, ymax) # type: ignore

                # If x axis is date, rotate tick labels and change font size
                if type(x_input.iloc[0]) == pd._libs.tslibs.timestamps.Timestamp: # type: ignore
                    ax[j,i].set_xticks(ax[j,i].get_xticks()) # type: ignore
                    ax[j,i].set_xticklabels(ax[j,i].get_xticklabels(), rotation=45, fontsize=8) # type: ignore

            plot_num += 1

    # Finish up plot
    plt.suptitle(main_title)
    plt.tight_layout()

    return plt