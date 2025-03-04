import numpy as np
import torch

import os

from mlproj_manager.file_management import read_json_file
from mlproj_manager.util.experiments_util import access_dict

from src.utils import aggregate_over_bins, plot_results, parse_plots_and_analysis_terminal_arguments

DEBUG = False
BIN_SIZE = {"test_accuracy_per_epoch": 100, "average_test_accuracy_per_epoch": 100, "ln_weight_magnitude": 1,
            "self_attention_weight_magnitude": 1, "mlp_weight_magnitude": 1, "network_parameter_magnitude": 1,
            "sa_weight_magnitude_median": 1, "mlp_weight_magnitude_median": 1,
            "last_in_projection_weight_magnitude": 1, "last_in_projection_weight_magnitude_median": 1,
            "sa_in_proj_weight_magnitude": 1, "sa_in_proj_weight_magnitude_median": 1}
AGG_FUNC = {"test_accuracy_per_epoch": "max", "average_test_accuracy_per_epoch": "max", "ln_weight_magnitude": "max",
            "self_attention_weight_magnitude": "max", "mlp_weight_magnitude": "max", "network_parameter_magnitude": "max",
            "sa_weight_magnitude_median": "max", "mlp_weight_magnitude_median": "max",
            "last_in_projection_weight_magnitude": "max", "last_in_projection_weight_magnitude_median": "max",
            "sa_in_proj_weight_magnitude": "max", "sa_in_proj_weight_magnitude_median": "max"}
WEIGHT_SUMMARY_NAMES = ["ln_weight_magnitude", "self_attention_weight_magnitude", "mlp_weight_magnitude",
                        "network_parameter_magnitude", "sa_weight_magnitude_median", "mlp_weight_magnitude_median",
                        "last_in_projection_weight_magnitude", "last_in_projection_weight_magnitude_median",
                        "sa_in_proj_weight_magnitude", "sa_in_proj_weight_magnitude_median"]


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


def compute_and_store_weight_magnitude_results(parameter_comb, results_dir):

    if DEBUG: print(f"\nParameter combination: {parameter_comb}")

    temp_results_dir = os.path.join(results_dir, parameter_comb)
    indices = np.load(os.path.join(temp_results_dir, "experiment_indices.npy"))
    if len(indices.shape) == 0: indices = indices.reshape(indices.size)
    indices.sort()
    store_frequency = 100
    total_number_of_epochs = 2000

    summary_dirs = [os.path.join(temp_results_dir, wm_dir) for wm_dir in WEIGHT_SUMMARY_NAMES]
    for d in summary_dirs: os.makedirs(d, exist_ok=True)

    for idx in indices:
        idx_weight_magnitude_lists = [[] for _ in range(len(WEIGHT_SUMMARY_NAMES))]

        for current_epoch in range(0, total_number_of_epochs + store_frequency, store_frequency):
            filename = f"index-{idx}_epoch-{current_epoch}.pt"
            try:
                temp_state_dict = torch.load(os.path.join(temp_results_dir, "model_parameters", filename), map_location="cpu")
            except EOFError:
                print(f"\n{filename = }\nParameter combination = {parameter_comb}"); raise EOFError

            # order of output: ln, sa, mlp, all, sa_median, mlp_median, last_in_proj, last_in_proj_median, sa_in_proj, sa_in_proj_median
            temp_summaries = compute_average_weight_magnitude(temp_state_dict)
            for i in range(len(temp_summaries)):
                idx_weight_magnitude_lists[i].append(temp_summaries[i])

        for i in range(len(WEIGHT_SUMMARY_NAMES)):
            np.save(os.path.join(summary_dirs[i], f"index-{idx}.npy"), np.array(idx_weight_magnitude_lists[i]))


def compute_average_weight_magnitude(state_dict: dict):

    (ln_sum, ln_numel, sa_sum, sa_numel, mlp_sum, mlp_numel, total_sum, total_numel, last_in_proj_sum,
     last_in_proj_numel, sa_in_proj_sum, sa_in_proj_numel) = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    sa_weights, mlp_weights, last_in_proj_weights, sa_in_proj_weights= [], [], [], []
    for n, p in state_dict.items():
        if (".cbp." in n) or (".redo." in n): continue
        is_weight = "weight" in n
        is_self_attention = ".self_attention." in n
        is_layer_norm = (".ln." in n) or (".ln_1." in n) or (".ln_2." in n)
        is_mlp_block = ".mlp." in n
        p_abs_sum, p_numel = p.abs().sum().item(), p.numel()
        if is_self_attention and is_weight:                     # weight magnitude of self-attention layers
            sa_sum += p_abs_sum
            sa_numel += p_numel
            sa_weights.extend(p.flatten().abs().tolist())
        if is_layer_norm and is_weight:                             # weight magnitude of layer norm layers
            ln_sum += p_abs_sum
            ln_numel += p_numel
        if is_mlp_block and is_weight:                              # weight magnitude of feed-forward layers in mlp block
            mlp_sum += p_abs_sum
            mlp_numel += p_numel
            mlp_weights.extend(p.flatten().abs().tolist())
        if "encoder_layer_7.self_attention.in_proj_weight" in n:    # weight of the last sa layer
            last_in_proj_sum += p_abs_sum
            last_in_proj_numel += p_numel
            last_in_proj_weights.extend(p.flatten().abs().tolist())
        if "self_attention.in_proj_weight" in n:
            sa_in_proj_sum += p_abs_sum
            sa_in_proj_numel += p_numel
            sa_in_proj_weights.extend(p.flatten().abs().tolist())
        total_sum += p_abs_sum                                      # weight magnitude of the entire network
        total_numel += p_numel

    return (ln_sum / ln_numel, sa_sum / sa_numel, mlp_sum / mlp_numel, total_sum / total_numel, np.median(sa_weights),
            np.median(mlp_weights), last_in_proj_sum / last_in_proj_numel, np.median(last_in_proj_weights),
            sa_in_proj_sum / sa_in_proj_numel, np.median(sa_in_proj_weights))


def print_average_test_accuracy(results_dict: dict):

    for i, (pc, temp_results) in enumerate(results_dict.items()):
        average = np.mean(temp_results)
        print(f"Parameter combination: {pc}\n\tAverage test accuracy = {average}")


def analyse_results(analysis_parameters: dict, save_plots: bool = True):

    results_dir = analysis_parameters["results_dir"]
    parameter_combinations = analysis_parameters["parameter_combinations"]
    summary_names = analysis_parameters["summary_names"]
    excluded_indices = access_dict(analysis_parameters, "excluded_indices", default={}, val_type=dict)
    compute_weight_magnitude_summaries = access_dict(analysis_parameters, "compute_weight_magnitude_summaries",
                                                     default=False, val_type=bool)
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
        elif sn in WEIGHT_SUMMARY_NAMES:
            if compute_weight_magnitude_summaries:
                for param_comb in parameter_combinations:
                    compute_and_store_weight_magnitude_results(param_comb, results_dir)
                compute_weight_magnitude_summaries = False
            results_data = get_results_data(results_dir, sn, parameter_combinations, excluded_indices=excluded_indices, max_samples=max_samples)
            plot_results(results_data, plot_parameters, plot_dir, sn, save_plots, plot_name_prefix)


if __name__ == "__main__":

    terminal_arguments = parse_plots_and_analysis_terminal_arguments()
    analysis_parameters = read_json_file(terminal_arguments.config_file)
    DEBUG = terminal_arguments.debug
    analyse_results(analysis_parameters, save_plots=terminal_arguments.save_plot)
