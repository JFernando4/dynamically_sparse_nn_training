import os
import numpy as np
import matplotlib.pyplot as plt

from mlproj_manager.util import access_dict

COLOR_DICT = {  # colorblind friendly according to nature
    "black":        "#000000",
    "orange":       "#e69f00",
    "sky_blue":     "#56b4e9",
    "blue_green":   "#009e73",
    "yellow":       "#f0e442",
    "blue":         "#0072b2",
    "vermilion":   "#d55e00",
    "pale_violet":  "#cc79a7",
    "gray":         "#808080"
}


def aggregate_over_bins(np_array: np.ndarray, bin_size: int, agg_func: str = "mean"):
    """
    Splits 1D-arrays into bins and aggregates over them. The array size must be divisible by bin_size.

    arguments:
        np_array (np.ndarray): 1D-array to aggregate over
        bin_size (int): size of the bins
        agg_func (str): name of aggregation function in ["mean", "max", "median", "min"]

    returns:
        1D array of size (np_array.size // bin_size)
    """

    if bin_size == 1:
        return np_array
    if (np_array.size % bin_size) != 0:
        raise ValueError(f"Size of np_array ({np_array.size}) is not divisible by bin_size ({bin_size}).")
    assert agg_func in ["mean", "max", "median", "min"]

    agg_func_dict = {"mean": np.average, "max": np.max, "min": np.min, "median": np.median}

    num_bins = np_array.size // bin_size
    reshaped_array = np_array.reshape(num_bins, bin_size)

    return agg_func_dict[agg_func](reshaped_array, axis=1)


def plot_results(results_data: dict, plot_parameters: dict, plot_dir: str, measurement_name: str,
                 save_plots: bool = True, plot_name_prefix: str = "", verbose: bool = True):
    """ Plots the data in results_data accoring to the parameters in plot_parameters """

    os.makedirs(plot_dir, exist_ok=True)

    color_order = access_dict(plot_parameters, "color_order", list(COLOR_DICT.keys()), list)
    alpha = access_dict(plot_parameters, "alpha", 0.1, float)
    x_label = access_dict(plot_parameters, "x_label", "Permutation Number", str)
    y_label = measurement_name if "y_label" not in plot_parameters.keys() else plot_parameters["y_label"]
    labels = access_dict(plot_parameters, "labels", list(results_data.keys()), list)
    linestyles = access_dict(plot_parameters, "linestyles", ["-"] * len(results_data), list)
    ylim = access_dict(plot_parameters, "ylim", None)
    yticks = access_dict(plot_parameters, "yticks", None)
    xlim = access_dict(plot_parameters, "xlim", None)
    xticks = access_dict(plot_parameters, "xticks", None)
    log_y_scale = access_dict(plot_parameters, "log_y_scale", False, val_type=bool)
    visible_grid = access_dict(plot_parameters, "visible_grid", default=True, val_type=bool)

    for i, (pc, temp_results) in enumerate(results_data.items()):

        average = np.mean(temp_results, axis=0)
        num_samples = temp_results.shape[0]
        if num_samples == 1:
            ste = np.zeros_like(average)
        else:
            ste = np.std(temp_results, axis=0, ddof=1) / np.sqrt(num_samples)
        if verbose:
            print(f"\t{pc}\n\tNumber of samples: {num_samples}")
            print(f"\tMax: {np.max(average):.5f}\n\tMin: {np.min(average):.5f}")
            print(f"\tMean: {average}")
        if num_samples < 15 and "_temp" not in measurement_name:
            measurement_name += "_temp"

        x_axis = np.arange(average.size)
        plt.plot(x_axis, average, label=labels[i], color=COLOR_DICT[color_order[i]], linestyle=linestyles[i])
        plt.fill_between(x_axis, average - ste, average + ste, color=COLOR_DICT[color_order[i]], alpha=alpha)

    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.legend()
    plt.grid(visible=visible_grid, axis="y")
    if log_y_scale:
        plt.yscale("log")
    if ylim is not None:
        plt.ylim(ylim)
    if xlim is not None:
        plt.xlim(xlim)
    if yticks is not None:
        plt.yticks(yticks)
    if xticks is not None:
        plt.xticks(xticks)

    if save_plots:
        plot_path = os.path.join(plot_dir, f"{plot_name_prefix}_{measurement_name}.svg")
        plt.savefig(plot_path, dpi=200)
    else:
        plt.show()
    plt.close()


def plot_avg_with_shaded_region(results_data: dict = None, results_avg: dict = None, results_low: dict = None,
                                results_high: dict = None, x_axis: dict = None, plot_parameters: dict = None,
                                plot_dir: str = "", measurement_name: str = "", save_plots: bool = True,
                                plot_name_prefix: str = "", num_samples: int = None):
    """ Plots the data in results_data accoring to the parameters in plot_parameters """
    assert (results_data is not None) or ((results_avg is not None) and (results_low is not None) and (results_high is not None))
    os.makedirs(plot_dir, exist_ok=True)

    color_order = access_dict(plot_parameters, "color_order", list(COLOR_DICT.keys()), list)
    alpha = access_dict(plot_parameters, "alpha", 0.1, float)
    x_label = access_dict(plot_parameters, "x_label", "Permutation Number", str)
    y_label = measurement_name if "y_label" not in plot_parameters.keys() else plot_parameters["y_label"]
    default_labels = results_data.keys() if results_data is not None else results_avg.keys()
    labels = access_dict(plot_parameters, "labels", list(default_labels), list)
    default_linestyles = ["-"] * len(results_data) if results_data is not None else ["-"] * len(results_avg)
    linestyles = access_dict(plot_parameters, "linestyles", default_linestyles, list)
    ylim = access_dict(plot_parameters, "ylim", None)
    yticks = access_dict(plot_parameters, "yticks", None)
    xlim = access_dict(plot_parameters, "xlim", None)
    visible_grid = access_dict(plot_parameters, "visible_grid", default=True, val_type=bool)

    results_dict = results_data if results_data is not None else results_avg
    for i, (pc, temp_results) in enumerate(results_dict.items()):
        if results_data is not None:
            temp_average = np.mean(temp_results, axis=0)
            num_samples = temp_results.shape[0]
            ste = np.std(temp_results, axis=0, ddof=1) / np.sqrt(num_samples)
            temp_low, temp_high = temp_average - ste, temp_average + ste
        else:
            temp_average, temp_low, temp_high = results_avg[pc], results_low[pc], results_high[pc]

        temp_x_axis = np.arange(temp_average.size) if x_axis is None else x_axis
        print(f"\t{pc}\n\tNumber of samples: {num_samples}")
        print(f"\tMax: {np.max(temp_average):.5f}\n\tMin: {np.min(temp_average):.5f}")

        plt.plot(temp_x_axis, temp_average, label=labels[i], color=COLOR_DICT[color_order[i]], linestyle=linestyles[i])
        plt.fill_between(temp_x_axis, temp_low, temp_high, color=COLOR_DICT[color_order[i]], alpha=alpha)

    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.legend()
    plt.grid(visible=visible_grid, axis="y")

    if ylim is not None:
        plt.ylim(ylim)
    if xlim is not None:
        plt.xlim(xlim)
    if yticks is not None:
        plt.yticks(yticks)

    if save_plots:
        plot_path = os.path.join(plot_dir, f"{plot_name_prefix}_{measurement_name}.svg")
        plt.savefig(plot_path, dpi=200)
    else:
        plt.show()
    plt.close()
