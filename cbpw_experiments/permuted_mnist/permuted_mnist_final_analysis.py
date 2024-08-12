import numpy
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import numpy as np
import pickle
import argparse

import os
import re
import torch
from mlproj_manager.plots_and_summaries.plotting_functions import line_plot_with_error_bars, lighten_color
from scipy.stats import pearsonr

from mlproj_manager.file_management import read_json_file, store_object_with_several_attempts
from mlproj_manager.util import access_dict
from mlproj_manager.problems.supervised_learning import MnistDataSet

from src.networks import ThreeHiddenLayerNetwork


COLOR_LIST = [  # colorblind friendly according to nature
    "#000000",  # black
    "#e69f00",  # orange
    "#56b4e9",  # sky blue
    "#009e73",  # blue green
    "#f0e442",  # yellow
    "#0072b2",  # blue
    "#d55e00",  # vermillion
    "#cc79a7"  # pale violet
]

COLOR_DICTT = {  # colorblind friendly according to nature
    "black":        "#000000",
    "orange":       "#e69f00",
    "sky_blue":     "#56b4e9",
    "blue_green":   "#009e73",
    "yellow":       "#f0e442",
    "blue":         "#0072b2",
    "vermillion":   "#d55e00",
    "pale_violet":  "#cc79a7"
}


DEBUG = False


def get_average_over_bins(np_array: np.ndarray, bin_size: int):
    """ Breaks array into bins and takes the average over the bins. Array must be divisible by bin_size. """
    if bin_size == 1:
        return np_array

    num_bins = np_array.size // bin_size
    reshaped_array = np_array.reshape(num_bins, bin_size)

    return np.average(reshaped_array, axis=1)

def compute_weight_magnitude_per_epoch(results_dir, recompute_results: bool = True):
    """
    Computes the average weight magnitudes for the model parameters stored for each index
    """

    weight_magnitude_dir = os.path.join(results_dir, "weight_magnitude_per_epoch")
    os.makedirs(weight_magnitude_dir, exist_ok=recompute_results)
    model_parameter_dir = os.path.join(results_dir, "model_parameters")

    placeholder_net = ThreeHiddenLayerNetwork(use_layer_norm="use_ln-True" in os.path.basename(results_dir),
                                              use_cbp="use_cbp-True" in os.path.basename(results_dir),
                                              replacement_rate=0.1, maturity_threshold=1)

    for k, index_name in enumerate(os.listdir(model_parameter_dir)):

        # load list of model parameters
        with open(os.path.join(model_parameter_dir, index_name), mode="rb") as temp_parameters_file:
            temp_model_parameters = pickle.load(temp_parameters_file)

        temp_weight_magnitudes = np.zeros(len(temp_model_parameters))

        # for each dictionary stored in model parameters, load parameters and compute avg weight magnitude
        for i in range(len(temp_model_parameters)):
            placeholder_net.load_state_dict(temp_model_parameters[i])

            weight_magnitude = 0.0
            total_weights = 0.0

            for p in placeholder_net.parameters():
                if p.requires_grad:
                    weight_magnitude += p.abs().sum()
                    total_weights += p.numel()

            temp_weight_magnitudes[i] += weight_magnitude / total_weights

        # store summaries
        temp_results_file = os.path.join(weight_magnitude_dir, f"index-{k}.npy")
        np.save(temp_results_file, temp_weight_magnitudes)


def handle_missing_measurement(results_dir: str, measurement_name: str, recompute_results=False):

    if measurement_name == "dead_units":
        pass
    elif measurement_name == "weight_magnitude_per_epoch":
        compute_weight_magnitude_per_epoch(results_dir, recompute_results=recompute_results)
    elif measurement_name in ["average_gradient_magnitude_per_checkpoint", "train_accuracy_per_checkpoint", "train_loss_per_checkpoint"]:
        return
    else:
        raise ValueError(f"{measurement_name} is not a valid measurement type.")


def get_results_data(results_dir: str, measurement_name: str, parameter_combination: list[str], bin_size=1):

    results = {}
    for pc in parameter_combination:
        temp_results_dir = os.path.join(results_dir, pc)
        measurement_dir = os.path.join(temp_results_dir, measurement_name)

        # if not os.path.exists(measurement_dir) or recompute_results:
        #     handle_missing_measurement(temp_results_dir, measurement_name, recompute_results=recompute_results)

        results[pc] = []
        for filename in os.listdir(measurement_dir):
            temp_measurement_array = np.load(os.path.join(measurement_dir, filename))
            results[pc].append(get_average_over_bins(temp_measurement_array, bin_size))
        results[pc] = np.array(results[pc])

    return results


def plot_results(results_data: dict, plot_parameters: dict, plot_dir: str, measurement_name: str,
                 save_plots: bool = True, plot_name_prefix: str = ""):
    """ Plots the data in results_data accoring to the parameters in plot_parameters """

    os.makedirs(plot_dir, exist_ok=True)

    color_order = access_dict(plot_parameters, "color_order", list(COLOR_DICTT.keys()), list)
    alpha = access_dict(plot_parameters, "alpha", 0.1, float)
    labels = access_dict(plot_parameters, "labels", list(results_data.keys()), list)
    linestyles = access_dict(plot_parameters, "linestyles", ["-"] * len(results_data), list)
    ylim = access_dict(plot_parameters, "ylim", None)
    yticks = access_dict(plot_parameters, "yticks", None)
    xlim = access_dict(plot_parameters, "xlim", None)

    for i, (pc, temp_results) in enumerate(results_data.items()):

        average = np.mean(temp_results, axis=0)
        ste = np.std(temp_results, axis=0, ddof=1) / np.sqrt(temp_results.shape[0])
        print(f"\t{pc}\n\tNumber of samples: {temp_results.shape[0]}")

        x_axis = np.arange(average.size)
        plt.plot(x_axis, average, label=labels[i], color=COLOR_DICTT[color_order[i]], linestyle=linestyles[i])
        plt.fill_between(x_axis, average - ste, average + ste, color=COLOR_DICTT[color_order[i]], alpha=alpha)

    plt.ylabel(measurement_name)
    plt.xlabel("Permutation Number")
    plt.legend()
    plt.grid(visible=True, axis="y")

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


def compute_difference_in_loss_after_reinitialization(results_dir: str, parameter_combinations: list[str]):
    """ Computes the difference in loss before and after reinitialization """

    loss_before = get_results_data(results_dir, "loss_before_topology_update", parameter_combinations)
    loss_after = get_results_data(results_dir, "loss_after_topology_update", parameter_combinations)

    for pc in parameter_combinations:
        print(f"\t{pc}")
        min_length = min(loss_before[pc].shape[1], loss_after[pc].shape[1])
        if DEBUG: print(f"\t{loss_before[pc].shape[1] = }, \t{loss_after[pc].shape[1] = } ")
        difference = loss_after[pc][:, :min_length] - loss_before[pc][:, :min_length]
        average_difference = np.average(difference, axis=1)
        total_average = np.average(average_difference)
        ste_average_difference = np.std(average_difference, ddof=1) / np.sqrt(average_difference.size)
        print(f"\t\ttAverage Difference: {total_average:.4f}")
        print(f"\t\tStandard Error of Difference: {ste_average_difference:.4f}")


def analyse_results(analysis_parameters: dict, save_plots: bool = True):

    results_dir = analysis_parameters["results_dir"]
    parameter_combinations = analysis_parameters["parameter_combinations"]
    summary_names = analysis_parameters["summary_names"]
    plot_dir = analysis_parameters["plot_dir"]
    bin_sizes = access_dict(analysis_parameters, "bin_sizes", default=[1] * len(summary_names), val_type=list)
    plot_parameters = access_dict(analysis_parameters, "plot_parameters", default={}, val_type=dict)
    plot_name_prefix = access_dict(analysis_parameters, "plot_name_prefix", default="", val_type=str)

    for sn, bs in zip(summary_names, bin_sizes):

        if sn in ["difference_in_loss_after_reinitialization"]:
            print(f"Summary name: {sn}")
            compute_difference_in_loss_after_reinitialization(results_dir, parameter_combinations)
        else:
            results_data = get_results_data(results_dir, sn, parameter_combinations, bin_size=bs)
            plot_results(results_data, plot_parameters, plot_dir, sn, save_plots, plot_name_prefix)


def parse_terminal_arguments():
    """ Reads experiment arguments """
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--config_file", action="store", type=str, required=True,
                                 help="JSON file with analysis configurations.")
    argument_parser.add_argument("--save_plot", action="store_true", default=False)
    argument_parser.add_argument("--debug", action="store_true", default=False)
    return argument_parser.parse_args()


if __name__ == "__main__":

    terminal_arguments = parse_terminal_arguments()
    analysis_parameters = read_json_file(terminal_arguments.config_file)

    DEBUG = terminal_arguments.debug

    analyse_results(analysis_parameters, save_plots=terminal_arguments.save_plot)
