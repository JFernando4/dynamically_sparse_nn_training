import numpy as np

import os

from mlproj_manager.file_management import read_json_file
from mlproj_manager.util.experiments_util import access_dict

from src.utils import aggregate_over_bins, plot_results, parse_plots_and_analysis_terminal_arguments

DEBUG = False
BIN_SIZE = {"test_accuracy_per_epoch": 100, "average_test_accuracy_per_epoch": 100}
AGG_FUNC = {"test_accuracy_per_epoch": "max", "average_test_accuracy_per_epoch": "max"}


def get_results_data(results_dir: str, measurement_name: str, parameter_combination: list[str],
                     excluded_indices: dict, max_samples: int = 15):

    results = {}
    for pc in parameter_combination:
        pc_excluded_indices = [] if pc not in excluded_indices.keys() else excluded_indices[pc]
        results[pc] = get_parameter_combination_results(pc, results_dir, measurement_name, pc_excluded_indices, max_samples)

    return results


def get_results_data_accuracy_diff(results_dir: str, parameter_combination: list[str], base_lines: list[str],
                                   excluded_indices: dict, max_samples: int = 15):

    results = {}
    for pc in parameter_combination:
        pc_excluded_indices = [] if pc not in excluded_indices.keys() else excluded_indices[pc]
        pc_results = get_parameter_combination_results(pc, results_dir, "test_accuracy_per_epoch", pc_excluded_indices, max_samples)
        baseline_results = get_parameter_combination_results(base_lines[pc], results_dir, "test_accuracy_per_epoch", pc_excluded_indices, max_samples)
        results[pc] = pc_results - baseline_results

    return results


def get_parameter_combination_results(parameter_comb, results_dir, measurement_name, pc_excluded_indices: list,
                                      max_samples: int = 15):
    assert measurement_name in BIN_SIZE.keys() and measurement_name in AGG_FUNC.keys()
    if DEBUG: print(f"\nParameter combination: {parameter_comb}")

    temp_results_dir = os.path.join(results_dir, parameter_comb)
    indices = np.load(os.path.join(temp_results_dir, "experiment_indices.npy"))
    if len(indices.shape) == 0: indices = indices.reshape(indices.size)
    indices.sort()
    measurement_dir = os.path.join(temp_results_dir, measurement_name)

    bin_size = BIN_SIZE[measurement_name]
    temp_results = []

    current_sample = 0
    for idx in indices:
        if current_sample >= max_samples: break
        if idx in pc_excluded_indices: continue

        filename = f"index-{idx}.npy"
        try:
            temp_measurement_array = np.load(os.path.join(measurement_dir, filename))
        except EOFError:
            if DEBUG:
                print(f"\n{filename = }\nParameter combination = {parameter_comb}\nMeasurement = {measurement_name}")
                print(f"\n{results_dir = }\n")
            raise EOFError
        temp_results.append(aggregate_over_bins(temp_measurement_array, bin_size, agg_func=AGG_FUNC[measurement_name]))

        if DEBUG: print(f"index: {idx}\tLast task performance: {temp_results[-1][-1]}")
        current_sample += 1

    return np.array(temp_results)


def print_average_test_accuracy(results_dict: dict):

    for i, (pc, temp_results) in enumerate(results_dict.items()):
        average = np.mean(temp_results)
        print(f"Parameter combination: {pc}\n\tAverage test accuracy = {average}")


def analyse_results(analysis_parameters: dict, save_plots: bool = True):

    results_dir = analysis_parameters["results_dir"]
    parameter_combinations = analysis_parameters["parameter_combinations"]
    summary_names = analysis_parameters["summary_names"]
    excluded_indices = access_dict(analysis_parameters, "excluded_indices", default={}, val_type=dict)
    max_samples = access_dict(analysis_parameters, "max_samples", default=15, val_type=int)
    base_lines = access_dict(analysis_parameters, "base_lines", default={}, val_type=dict)
    plot_dir = access_dict(analysis_parameters, "plot_dir", default="")
    plot_parameters = access_dict(analysis_parameters, "plot_parameters", default={}, val_type=dict)
    plot_name_prefix = access_dict(analysis_parameters, "plot_name_prefix", default="", val_type=str)

    for sn in summary_names:

        if sn == "test_accuracy_per_epoch":
            results_data = get_results_data(results_dir, sn, parameter_combinations, excluded_indices, max_samples)
            plot_results(results_data, plot_parameters, plot_dir, sn, save_plots, plot_name_prefix)
        elif sn == "average_test_accuracy_per_epoch":
            results_data = get_results_data(results_dir, "test_accuracy_per_epoch", parameter_combinations, max_samples)
            print_average_test_accuracy(results_data)
        elif sn == "test_accuracy_with_baseline":
            results_data = get_results_data_accuracy_diff(results_dir, parameter_combinations, base_lines, excluded_indices, max_samples)
            plot_results(results_data, plot_parameters, plot_dir, sn, save_plots, plot_name_prefix)


if __name__ == "__main__":

    terminal_arguments = parse_plots_and_analysis_terminal_arguments()
    analysis_parameters = read_json_file(terminal_arguments.config_file)
    DEBUG = terminal_arguments.debug
    analyse_results(analysis_parameters, save_plots=terminal_arguments.save_plot)
