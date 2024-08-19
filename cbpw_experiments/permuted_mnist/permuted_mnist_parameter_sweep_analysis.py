import numpy as np
import argparse

import os

from mlproj_manager.file_management import read_json_file


DEBUG = False

def check_conversion(s: str):
    """
    Checks if the give string can be converted to int or float and returns the int or float value if possible
    otherwise it returns the original string
    """

    # Check if the string can be converted to an int
    try:
        int_val = int(s)
        return int_val
    except ValueError:
        pass

    # Check if the string can be converted to a float
    try:
        float_val = float(s)
        return float_val
    except ValueError:
        pass

    return s


def training_accuracy_list_format(results_dir: str):

    """ Prints the training accuracy for each parameter combination in results dir """
    for param_comb in os.listdir(results_dir):

        temp_dir = os.path.join(results_dir, param_comb, "train_accuracy_per_checkpoint")
        if not os.path.isdir(temp_dir): continue
        no_results = len(os.listdir(temp_dir)) == 0
        if no_results: continue
        indices = np.load(os.path.join(results_dir, param_comb, "experiment_indices.npy"))
        num_samples = indices.size
        print(f"{param_comb}\t\t\tSamples: {num_samples}")
        results = []

        for idx in indices:
            file_name = f"index-{idx}.npy"
            try:
                results_file_name = os.path.join(temp_dir, file_name)
                results.append(np.load(results_file_name))
            except EOFError:
                print(f"Index: {idx}, Parameter combination: {param_comb}")
                raise EOFError

        mean = np.round(np.average(results), 4)
        std_error = np.round(np.std(np.average(results, axis=1), ddof=1) / np.sqrt(num_samples), 4)
        print(f"Avg: {mean:.4f}\tSE: {std_error:.4f}")


def training_accuracy_table_format(results_dir: str, column_var: str, row_var: str, grow_method: str = None,
                                   prune_method: str = None):
    """
    Prints the training accuracy for each parameter combination in results as a table
    """

    column_var_list, row_var_list = get_sorted_values(os.listdir(results_dir), column_var, row_var)

    average_results, num_samples, max_acc_indices = compute_average_training_accuracy_for_table(column_var_list,
                                                                                                row_var_list,
                                                                                                results_dir,
                                                                                                column_var, row_var,
                                                                                                grow_method=grow_method,
                                                                                                prune_method=prune_method)

    print_table(average_results, num_samples, max_acc_indices, column_var_list, row_var_list)


def get_sorted_values(parameter_combinations: list[str], column_var: str, row_var: str):
    column_var_list = []
    row_var_list = []

    for param_comb in parameter_combinations:
        if column_var not in param_comb: continue
        if row_var not in param_comb: continue

        # param_comb is formatted as "column_var-val1_row_var-val2_other_var-val3"
        # the line below gets the value immediately after the given variable
        temp_cv_value = param_comb.split(column_var)[1].split("_")[0][1:]
        temp_cv_value = check_conversion(temp_cv_value)

        temp_rv_value = param_comb.split(row_var)[1].split("_")[0][1:]
        temp_rv_value = check_conversion(temp_rv_value)

        if temp_cv_value not in column_var_list:
            column_var_list.append(temp_cv_value)
        if temp_rv_value not in row_var_list:
            row_var_list.append(temp_rv_value)

    column_var_list.sort()
    row_var_list.sort()
    print(f"Column variable values: {column_var_list}")
    print(f"Row variable values: {row_var_list}")
    return column_var_list, row_var_list


def compute_average_training_accuracy_for_table(column_var_list: list, row_var_list: list, results_dir: str,
                                                column_var: str, row_var: str, grow_method: str = None,
                                                prune_method: str = None):

    average_results = np.zeros((len(column_var_list), len(row_var_list))) + np.nan
    num_samples = np.zeros((len(column_var_list), len(row_var_list)), dtype=np.int32)

    base_name = os.listdir(results_dir)[0]
    if grow_method is not None and prune_method is not None:
        base_name = insert_column_and_row_values(base_name, "grow_method", "prune_method", (grow_method, prune_method))
        print(f"{prune_method = }")
        print(f"{grow_method = }")

    max_acc = -np.inf
    max_acc_indices = (-1, -1)
    for i, cv in enumerate(column_var_list):
        for j, rv in enumerate(row_var_list):

            param_comb_name = insert_column_and_row_values(base_name, column_var, row_var, (cv, rv))
            if DEBUG: print(param_comb_name)
            temp_dir = os.path.join(results_dir, param_comb_name, "train_accuracy_per_checkpoint")

            if not os.path.isdir(temp_dir): continue
            indices = np.load(os.path.join(results_dir, param_comb_name, "experiment_indices.npy"))
            num_samples[i, j] = indices.size
            results = []

            for idx in indices:
                file_name = f"index-{idx}.npy"
                try:
                    results_file_name = os.path.join(temp_dir, file_name)
                    results.append(np.load(results_file_name))
                except EOFError:
                    print(f"Index: {idx}, Parameter combination: {param_comb_name}")
                    raise EOFError

            average_results[i, j] = np.average(results)
            if average_results[i, j] > max_acc:
                max_acc = average_results[i, j]
                max_acc_indices = (i, j)

    return average_results, num_samples, max_acc_indices


def insert_column_and_row_values(base_name: str, column_var: str, row_var: str, values: tuple):
    """
    Inserts the row and column values into the base nam
    This could be done more cleanly using regular expressions, but I'm lazy
    """

    cv, rv = values

    split_list = base_name.split("-")

    for i, part in enumerate(split_list):   # quick hack for when parameter values are 1e-05, 1e-10, etc
        if "1e" in part:
            split_list.pop(i)
            break
    if DEBUG: print(split_list)

    new_name = []
    column_correct_next = False
    row_correct_next = False
    for i, part in enumerate(split_list):
        temp_part = part
        if row_correct_next or column_correct_next:
            temp_part_split = temp_part.split("_")
            temp_part_split[0] = str(rv) if row_correct_next else str(cv)

            if len(temp_part_split) >= 2:
                if temp_part_split[1] == "normal":  # quick hack for when parameter values is "kaiming_normal"
                    temp_part_split.pop(1)

            temp_part = "_".join(temp_part_split)
            column_correct_next = False
            row_correct_next = False
        if column_var in part:
            column_correct_next = True
        if row_var in part:
            row_correct_next = True
        new_name.append(temp_part)
    return "-".join(new_name)


def print_table(average_results: np.ndarray, num_samples: np.ndarray, max_acc_indices: tuple[int, int],
                column_var_list: list, row_var_list: list):

    print("", end="\t|   ")
    space_after = 1
    for cv in column_var_list:
        space = " " * (12 + space_after - len(f"{cv}"))
        print(f"{cv}", end=f"{space}|\t")
    print("")
    for j, rv in enumerate(row_var_list):
        print(f"{rv}", end="\t|   ")
        for i, cv in enumerate(column_var_list):
            best_indicator = " "
            if i == max_acc_indices[0] and j == max_acc_indices[1]:
                best_indicator = "*"
            print(f"{average_results[i, j]:.4f} ({num_samples[i, j]}){best_indicator}", end=" " * space_after + "|\t")
        print("")

def analyse_results(analysis_parameters: dict):

    results_dir = analysis_parameters["results_dir"]
    display_format = analysis_parameters["display_format"]

    if display_format == "list":
        training_accuracy_list_format(results_dir)
    elif display_format == "table":
        assert "column_var" in analysis_parameters.keys()
        assert "row_var" in analysis_parameters.keys()
        grow_method = None if "grow_method" not in analysis_parameters.keys() else analysis_parameters["grow_method"]
        prune_method = None if "prune_method" not in analysis_parameters.keys() else analysis_parameters["prune_method"]
        training_accuracy_table_format(results_dir, analysis_parameters["column_var"], analysis_parameters["row_var"],
                                       grow_method, prune_method)
    else:
        raise ValueError(f"{display_format} is not a valid display format.")


def parse_terminal_arguments():
    """ Reads experiment arguments """
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--config_file", action="store", type=str, required=True,
                                 help="JSON file with analysis configurations.")
    argument_parser.add_argument("--grow_method", action="store", type=str, required=False, default=None,
                                 help="Grow method for selective weight reinitialization.")
    argument_parser.add_argument("--prune_method", action="store", required=False, default=None,
                                 help="Prune method for selective weight reinitialization.")
    argument_parser.add_argument("--debug", action="store_true", default=False)
    return argument_parser.parse_args()


if __name__ == "__main__":

    terminal_arguments = parse_terminal_arguments()
    analysis_parameters = read_json_file(terminal_arguments.config_file)
    analysis_parameters["grow_method"] = terminal_arguments.grow_method
    analysis_parameters["prune_method"] = terminal_arguments.prune_method
    DEBUG = terminal_arguments.debug
    analyse_results(analysis_parameters)
