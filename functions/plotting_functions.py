import matplotlib.pyplot as plt

def two_panel_histogram(
        data: List[float],
        bins: int,
        main_title: str,
        linear_plot_x_label: str,
        linear_plot_y_label: str,
        log10_plot_x_label: str,
        log10_plot_y_label: str,
        fig_x_dim = 8,
        fig_y_dim = 4
):

    fig, ax = plt.subplots(1, 2, figsize=(fig_x_dim, fig_y_dim))

    ax[0].hist(data, bins=bins)
    ax[0].set_xlabel(linear_plot_x_label)
    ax[0].set_ylabel(linear_plot_y_label)

    ax[1].hist(np.log10(data), bins=bins)
    ax[1].set_xlabel(log10_plot_x_label)
    ax[1].set_ylabel(log10_plot_y_label)

    plt.suptitle(main_title)
    plt.tight_layout()

    return plt