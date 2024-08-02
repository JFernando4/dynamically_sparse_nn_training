import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import numpy as np
import pickle
import argparse

import os
import torch
from mlproj_manager.plots_and_summaries.plotting_functions import line_plot_with_error_bars, lighten_color
from scipy.stats import pearsonr

from mlproj_manager.file_management import read_json_file
from mlproj_manager.problems.supervised_learning import MnistDataSet

from src.networks import ThreeHiddenLayerNetwork


def get_average_over_bins(np_array, bin_size: int):

    num_bins = np_array.size // bin_size
    reshaped_array = np_array.reshape(num_bins, bin_size)
    return np.average(reshaped_array, axis=1)


def get_network_weight_magnitude(net: torch.nn.Module):
    pass


def get_network_average_activation(net: torch.nn.Module, test_data: DataLoader):
    pass


def get_average_measurement_per_checkpoint(results_dir: str, measurement_name: str):
    """
    :param results_dir: path to results directory
    :param measurement_name: one of these three ["train_accuracy", "train_loss", "average_gradient_magnitude"]
    """
    pass


def simple_training_accuracy_analysis(results_dir: str):

    """ Prints the training accuracy for each parameter combination in results dir """
    for param_comb in os.listdir(results_dir):

        temp_dir = os.path.join(results_dir, param_comb, "train_accuracy_per_checkpoint")
        if not os.path.isdir(temp_dir): continue
        no_results = len(os.listdir(temp_dir)) == 0
        if no_results: continue
        num_samples = len(os.listdir(temp_dir))
        print(f"{param_comb}\t\tSamples: {num_samples}")
        results = []

        for file_name in os.listdir(temp_dir):
            results_file_name = os.path.join(temp_dir, file_name)
            results.append(np.load(results_file_name))

        print(f"Mean: {np.round(np.average(results), 4)}\t"
              f"Standard Error: {np.round(np.std(np.average(results, axis=1), ddof=1) / np.sqrt(num_samples), 4)}")


def analyse_results(results_dir: str):

    bin_size = 200

    simple_training_accuracy_analysis(results_dir)


def parse_terminal_arguments():
    """ Reads experiment arguments """
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--results_dir_path", action="store", type=str, required=True,
                                 help="Path where results are.")
    # argument_parser.add_argument("--analysis_config_file", action="store", type=str, required=True,
    #                              help="JSON file with analysis configurations.")
    argument_parser.add_argument("--verbose", action="store_true", default=False)
    return argument_parser.parse_args()


if __name__ == "__main__":

    terminal_arguments = parse_terminal_arguments()

    results_dir = terminal_arguments.results_dir_path
    # analysis_parameters = read_json_file(terminal_arguments.analysis_config_file)
    analyse_results(results_dir)
