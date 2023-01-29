import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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