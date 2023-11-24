# built-in libraries
import time
import os
import argparse

# third party libraries
import torch
from torch.utils.data import DataLoader
import numpy as np
from torchvision import transforms
from scipy.linalg import svd

# from ml project manager
from mlproj_manager.problems import CifarDataSet
from mlproj_manager.experiments import Experiment
from mlproj_manager.util import turn_off_debugging_processes, get_random_seeds, access_dict
from mlproj_manager.util.data_preprocessing_and_transformations import ToTensor, Normalize

from src import build_resnet18
from src.networks.torchvision_modified_resnet import ResNet


# -------------------- For loading data and network parameters -------------------- #
def load_model_parameters(parameter_dir_path: str, index: int, epoch_number:int):
    """
    Loads the model parameters stored in parameter_dir_path corresponding to the index and epoch number
    return: torch module state dictionary
    """

    model_parameters_file_name = "index-{0}_epoch-{1}.pt".format(index, epoch_number)
    model_parameters_file_path = os.path.join(parameter_dir_path, model_parameters_file_name)

    if not os.path.isfile(model_parameters_file_path):
        error_message = "Couldn't find model parameters for index {0} and epoch number {1}.".format(index, epoch_number)
        raise ValueError(error_message)

    return torch.load(model_parameters_file_path)


def load_classes(classes_dir_path: str, index: int):
    """
    Loads the list of ordered classes used for partitioning the datta during the experiment
    return: list
    """

    classes_file_name = "index-{0}.npy".format(index)
    classes_file_path = os.path.join(classes_dir_path, classes_file_name)

    if not os.path.isfile(classes_file_path):
        error_message = "Couldn't find list of classes for index {0}.".format(index)
        raise ValueError(error_message)

    return np.load(classes_file_path)


def load_cifar_data(data_path: str) -> (CifarDataSet, DataLoader):
    """
    Loads the cifar 100 data set with normalization
    :param data_path: path to the directory containing the data set
    :return: torch DataLoader object
    """
    cifar_data = CifarDataSet(root_dir=data_path,
                              train=True,
                              cifar_type=100,
                              device=None,
                              image_normalization="max",
                              label_preprocessing="one-hot",
                              use_torch=True)

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    transformations = [
        ToTensor(swap_color_axis=True),  # reshape to (C x H x W)
        Normalize(mean=mean, std=std),  # center by mean and divide by std
    ]

    cifar_data.set_transformation(transforms.Compose(transformations))

    num_workers = 12
    batch_size = 1000
    dataloader = DataLoader(cifar_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return cifar_data, dataloader


# -------------------- For computing analysis of the network -------------------- #
@torch.no_grad()
def compute_average_weight_magnitude(net: ResNet):
    """ Computes average magnitude of the weights in the network """

    num_weights = 0
    sum_weight_magnitude = torch.tensor(0.0, device=net.fc.weight.device)

    for p in net.parameters():
        num_weights += p.numel()
        sum_weight_magnitude += torch.sum(torch.abs(p))

    return sum_weight_magnitude.cpu().item() / num_weights


@torch.no_grad()
def compute_dormant_units_proportion(net: ResNet, cifar_data_loader: DataLoader, dormant_unit_threshold: float = 0.01):
    """
    Computes the proportion of dormant units in a ResNet. It also returns the features of the last layer for the first
    1000 samples
    """

    device = net.fc.weight.device
    features_per_layer = []
    last_layer_activations = None
    num_samples = 1000

    for i, sample in enumerate(cifar_data_loader):
        image = sample["image"].to(device)
        temp_features = []
        net.forward(image, temp_features)

        features_per_layer = temp_features
        last_layer_activations = temp_features[-1].cpu()
        break

    dead_neurons = torch.zeros(len(features_per_layer), dtype=torch.float32)
    for layer_idx in range(len(features_per_layer) - 1):
        dead_neurons[layer_idx] = ((features_per_layer[layer_idx] != 0).float().mean(dim=(0, 2, 3)) < dormant_unit_threshold).sum()
    dead_neurons[-1] = ((features_per_layer[-1] != 0).float().mean(dim=0) < dormant_unit_threshold).sum()
    number_of_features = torch.sum(torch.tensor([layer_feats.shape[1] for layer_feats in features_per_layer])).item()
    return dead_neurons.sum().item() / number_of_features, last_layer_activations.numpy()


def compute_effective_rank(last_layer_features: np.ndarray):
    """ Computes the effective rank of the representation layer """

    singular_values = svd(last_layer_features, compute_uv=False, lapack_driver="gesvd")
    norm_sv = singular_values / np.sum(np.abs(singular_values))
    entropy = 0.0
    for p in norm_sv:
        if p > 0.0:
            entropy -= p * np.log(p)

    return np.e ** entropy


def compute_stable_rank(last_layer_features: np.ndarray):

    singular_values = svd(last_layer_features, compute_uv=False, lapack_driver="gesvd")
    sorted_singular_values = np.flip(np.sort(singular_values))
    cumsum_sorted_singular_values = np.cumsum(sorted_singular_values) / np.sum(singular_values)
    return np.sum(cumsum_sorted_singular_values < 0.99) + 1


def analyze_results(results_dir: str, data_path: str, dormant_unit_threshold: float = 0.01):
    """
    Analyses the parameters of a run and creates files with the results of the analysis
    :param results_dir: path to directory containing the results for a parameter combination
    :param dormant_unit_threshold: hidden units whose activation fall bellow this threshold are considered dormant
    """

    parameter_dir_path = os.path.join(results_dir, "model_parameters")
    experiment_indices_file_path = os.path.join(results_dir, "experiment_indices.npy")
    class_order_dir_path = os.path.join(results_dir, "class_order")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    number_of_epochs = np.arange(21) * 200  # by design the model parameters where store after each of these epochs)
    classes_per_task = 5                    # by design each task increases the data set by 5 classes
    experiment_indices = np.load(experiment_indices_file_path)

    net = build_resnet18(num_classes=100, norm_layer=torch.nn.BatchNorm2d)
    net.to(device)
    cifar_data, cifar_data_loader = load_cifar_data(data_path)

    for exp_index in experiment_indices:

        ordered_classes = load_classes(class_order_dir_path, index=exp_index)

        average_weight_magnitude_per_epoch = np.zeros(number_of_epochs.size - 1, dtype=np.float32)
        dormant_units_prop_before = np.zeros_like(average_weight_magnitude_per_epoch)
        effective_rank_before = np.zeros_like(average_weight_magnitude_per_epoch)
        stable_rank_before = np.zeros_like(average_weight_magnitude_per_epoch)
        dormant_units_prop_after = np.zeros_like(average_weight_magnitude_per_epoch)
        effective_rank_after = np.zeros_like(average_weight_magnitude_per_epoch)
        stable_rank_after = np.zeros_like(average_weight_magnitude_per_epoch)

        for i, epoch_number in enumerate(number_of_epochs):

            # get model parameters from before training on the task
            model_parameters = load_model_parameters(parameter_dir_path, index=exp_index, epoch_number=epoch_number)
            net.load_state_dict(model_parameters)

            # compute average weight magnitude
            average_weight_magnitude_per_epoch[i] = compute_average_weight_magnitude(net)

            # compute summaries for next task
            current_classes = ordered_classes[(i * classes_per_task):((i + 1) * classes_per_task)]
            cifar_data.select_new_partition(current_classes)

            prop_dormant, last_layer_features = compute_dormant_units_proportion(net, cifar_data_loader, dormant_unit_threshold)
            dormant_units_prop_after[i] = prop_dormant
            effective_rank_after[i] = compute_effective_rank(last_layer_features)
            stable_rank_after[i] = compute_stable_rank(last_layer_features)

            # compute summaries from data from previous tasks
            if i == 0: continue
            current_classes = ordered_classes[:(i * classes_per_task)]
            cifar_data.select_new_partition(current_classes)
            prop_dormant, last_layer_features = compute_dormant_units_proportion(net, cifar_data_loader, dormant_unit_threshold)

            dormant_units_prop_before[i] = prop_dormant
            effective_rank_before[i] = compute_effective_rank(last_layer_features)
            stable_rank_before[i] = compute_stable_rank(last_layer_features)

        print("Experiment index: {0}".format(exp_index))
        print("Average weight magnitude:\n\t{0}".format(average_weight_magnitude_per_epoch))
        print("Proportion of dead units on previous tasks:\n\t{0}".format(dormant_units_prop_before))
        print("Effective Rank on previous tasks:\nt\t{0}".format(effective_rank_before))
        print("Stable Rank on previous tasks:\nt\t{0}".format(stable_rank_before))
        print("Proportion of dead units on next task:\n\t{0}".format(dormant_units_prop_after))
        print("Effective Rank on next task:\nt\t{0}".format(effective_rank_after))
        print("Stable Rank on next task:\nt\t{0}".format(stable_rank_after))


def parse_arguments() -> dict:
    file_description = "Script for computing the effective rankk, number of dead neurons, and weight magnitude of the" \
                       " models trained during the incremental cifar experiment."
    parser = argparse.ArgumentParser(description=file_description)

    parser.add_argument('--results_dir', action="store", type=str, required=True,
                        help="Path to directory with the results of a parameter combination.")
    parser.add_argument('--data_path', action="store", type=str, required=True,
                        help="Path to directory with the CIFAR 100 data set.")
    parser.add_argument('--dormant_unit_threshold', action="store", type=float, default=0.01,
                        help="Units whose activations are less than this threshold are considered dormant.")

    args = parser.parse_args()
    return vars(args)


def main():

    analysis_arguments = parse_arguments()

    initial_time = time.perf_counter()
    analyze_results(results_dir=analysis_arguments["results_dir"],
                    data_path=analysis_arguments["data_path"],
                    dormant_unit_threshold=analysis_arguments["dormant_unit_threshold"])
    final_time = time.perf_counter()
    print("The running time in minutes is: {0:.2f}".format((final_time - initial_time) / 60))


if __name__ == "__main__":
    main()