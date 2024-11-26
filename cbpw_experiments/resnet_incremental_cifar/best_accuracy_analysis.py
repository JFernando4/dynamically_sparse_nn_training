import numpy as np
import argparse
import matplotlib.pyplot as plt

import os

from mlproj_manager.file_management import read_json_file
from mlproj_manager.util.experiments_util import access_dict

from src.utils import COLOR_DICT, aggregate_over_bins, plot_results

DEBUG = False
BIN_SIZE = {"test_accuracy_per_epoch": 200}


def get_results_data(results_dir: str, measurement_name: str, parameter_combination: list[str]):

    valid_measurements = BIN_SIZE.keys()
    assert measurement_name in valid_measurements
    bin_size = BIN_SIZE[measurement_name]

    results = {}
    for pc in parameter_combination:
        temp_results_dir = os.path.join(results_dir, pc)
        indices = np.load(os.path.join(temp_results_dir, "experiment_indices.npy"))
        measurement_dir = os.path.join(temp_results_dir, measurement_name)

        results[pc] = []
        for idx in indices:
            filename = f"index-{idx}.npy"
            try:
                temp_measurement_array = np.load(os.path.join(measurement_dir, filename))
            except EOFError:
                if DEBUG:
                    print(f"\n{filename = }\nParameter combination = {pc}\nMeasurement = {measurement_name}")
                    print(f"\n{results_dir = }\n")
                raise EOFError
            results[pc].append(aggregate_over_bins(temp_measurement_array, bin_size, agg_func="max"))
        results[pc] = np.array(results[pc])

    return results


def analyse_results(analysis_parameters: dict, save_plots: bool = True):

    results_dir = analysis_parameters["results_dir"]
    parameter_combinations = analysis_parameters["parameter_combinations"]
    summary_names = analysis_parameters["summary_names"]
    plot_dir = access_dict(analysis_parameters, "plot_dir", default="")
    plot_parameters = access_dict(analysis_parameters, "plot_parameters", default={}, val_type=dict)
    plot_name_prefix = access_dict(analysis_parameters, "plot_name_prefix", default="", val_type=str)

    for sn, bs in summary_names:
        results_data = get_results_data(results_dir, sn, parameter_combinations)
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
