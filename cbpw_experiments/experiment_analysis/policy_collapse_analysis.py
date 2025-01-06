import numpy as np
import argparse

import os

from mlproj_manager.file_management import read_json_file
from mlproj_manager.util.experiments_util import access_dict

from src.utils import aggregate_over_bins, plot_avg_with_shaded_region, bootstrapped_return, parse_plots_and_analysis_terminal_arguments

DEBUG = False
BIN_SIZE = {"average_return": 100000, "return_per_episode": 1, "termination_steps": 1}


def get_results_data(results_dir: str, measurement_name: str, parameter_combination: list[str]):

    valid_measurements = BIN_SIZE.keys()
    assert measurement_name in valid_measurements
    bin_size = BIN_SIZE[measurement_name]

    results = {}
    for pc in parameter_combination:
        temp_results_dir = os.path.join(results_dir, pc)
        indices = np.load(os.path.join(temp_results_dir, "experiment_indices.npy"))
        if len(indices.shape) == 0:
            indices = indices.reshape(indices.size)
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
            results[pc].append(aggregate_over_bins(temp_measurement_array, bin_size, agg_func="mean"))
        if measurement_name not in ["return_per_episode", "termination_steps"]:
            results[pc] = np.array(results[pc])

    return results


def compute_average_return_statistics(results_dir: str, parameter_combinations: list):
    return_per_episode = get_results_data(results_dir, "return_per_episode", parameter_combinations)
    episode_length = get_results_data(results_dir, "termination_steps", parameter_combinations)
    total_steps = analysis_parameters["total_steps"]

    x_axis, avg_return_per_episode, confidence_interval_low, confidence_interva_high = None, {}, {}, {}
    num_samples = len(return_per_episode[list(return_per_episode.keys())[0]])
    for pc in return_per_episode.keys():
        x_axis, temp_avg, _, _, temp_ci_low, temp_ci_high = bootstrapped_return(episode_length[pc],
                                                                                return_per_episode[pc],
                                                                                bin_size=BIN_SIZE["average_return"],
                                                                                total_steps=total_steps)
        avg_return_per_episode[pc] = temp_avg
        confidence_interval_low[pc], confidence_interva_high[pc] = temp_ci_low, temp_ci_high

    return x_axis, avg_return_per_episode, confidence_interval_low, confidence_interva_high, num_samples


def analyse_results(analysis_parameters: dict, save_plots: bool = True):

    results_dir = analysis_parameters["results_dir"]
    parameter_combinations = analysis_parameters["parameter_combinations"]
    summary_names = analysis_parameters["summary_names"]
    plot_dir = access_dict(analysis_parameters, "plot_dir", default="")
    plot_parameters = access_dict(analysis_parameters, "plot_parameters", default={}, val_type=dict)
    plot_name_prefix = access_dict(analysis_parameters, "plot_name_prefix", default="", val_type=str)

    for sn in summary_names:
        plot_args = {"plot_parameters": plot_parameters, "plot_dir": plot_dir, "measurement_name": sn,
                     "save_plots": save_plots, "plot_name_prefix": plot_name_prefix}
        if sn == "average_return":
            x_axis, avg_return, ci_low, ci_high, num_samples = compute_average_return_statistics(results_dir, parameter_combinations)
            plot_avg_with_shaded_region(results_avg=avg_return, results_low=ci_low, results_high=ci_high,
                                        x_axis=x_axis, num_samples=num_samples, **plot_args)
        elif sn == "average_return_over_run":
            results_data = get_results_data(results_dir, "return_per_episode", parameter_combinations)
            for k, v in results_data.items():
                print(f"Parameter combinations: {k}")
                num_runs = 0
                average_return = 0.0
                for run_array in v:
                    num_runs += 1
                    average_return += np.average(run_array)
                average_return /= num_runs
                print(f"\tAverage return over entire experiment: {average_return}\tSample size: {num_runs}")
        else:
            results_data = get_results_data(results_dir, sn, parameter_combinations)
            plot_avg_with_shaded_region(results_data=results_data, **plot_args)


if __name__ == "__main__":

    terminal_arguments = parse_plots_and_analysis_terminal_arguments()
    analysis_parameters = read_json_file(terminal_arguments.config_file)
    DEBUG = terminal_arguments.debug
    analyse_results(analysis_parameters, save_plots=terminal_arguments.save_plot)
