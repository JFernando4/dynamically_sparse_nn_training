# third party libraries
import torch
from torch.utils.data import DataLoader

# from src
from src.networks import ThreeHiddenLayerNetwork
from src.utils.evaluation_functions import compute_matrix_rank_summaries


@torch.no_grad()
def compute_dead_units_prop_and_stable_rank(net: ThreeHiddenLayerNetwork, data_loader: DataLoader, num_activations: int,
                                            batch_size: int = 30, num_mini_batches: int = 50):
    """
    Computes the proportion of dead units and the stable rank of the representation (the last layer)
    """
    num_inputs = 784
    num_layers = 3
    total_num_units = num_layers * num_activations

    # compute some number of activations
    all_activations = torch.zeros((num_mini_batches * batch_size, num_layers, num_activations), dtype=torch.float32)
    for i, sample in enumerate(data_loader):
        if i >= num_mini_batches:
            break
        image = sample["image"].reshape(batch_size, num_inputs)
        temp_acts = []
        net.forward(image, activations=temp_acts)
        for l in range(num_layers):
            all_activations[i * batch_size:(i + 1) * batch_size, :, l] = temp_acts[l]

    prop_dead_units = torch.sum((all_activations.sum(0) == 0.0)).item() / total_num_units
    stable_rank = compute_matrix_rank_summaries(all_activations[:, :, -1], prop=0.99, use_scipy=False)
    return prop_dead_units, stable_rank


@torch.no_grad()
def compute_dead_units_proportion(net: ThreeHiddenLayerNetwork, data_loader: DataLoader, num_activations: int = 10,
                                  batch_size: int = 30, num_inputs: int = 784,  num_mini_batches: int = 50):
    """ computes the proportion of dead units in the network"""

    num_layers = 3
    all_activations = torch.zeros((num_mini_batches * batch_size, num_layers * num_activations), dtype=torch.float32)
    for i, sample in enumerate(data_loader):

        if i >= num_mini_batches:
            break

        image = sample["image"].reshape(batch_size, num_inputs)
        temp_acts = []
        net.forward(image, activations=temp_acts)

        stacked_act = torch.hstack(temp_acts)
        all_activations[i * batch_size:(i+1) * batch_size, :] = stacked_act

    sum_activations = all_activations.sum(0)
    return torch.mean((sum_activations == 0.0).to(torch.float32))


def initialize_results_dict(
        steps_per_task: int,
        num_permutations: int,
        running_avg_window: int,
        batch_size: int,
        device: torch.device,
        use_swr: bool = False,
        topology_update_freq: int = 1,
        use_redo: bool = False,
        use_cbp: bool = False,
        use_ln: bool = False,
        extended_summaries: bool = False
) -> dict:
    """
    Initializes the results dictionary for the permuted mnist experiment
    """
    results_dict = {}
    defaults = {"device": device, "dtype": torch.float32}

    total_ckpts = steps_per_task * num_permutations // (running_avg_window * batch_size)
    results_dict["train_loss_per_checkpoint"] = torch.zeros(total_ckpts, **defaults)
    results_dict["train_accuracy_per_checkpoint"] = torch.zeros(total_ckpts, **defaults)

    if use_swr or use_redo or use_cbp:
        results_dict["num_replaced"] = []
        if use_swr:
            total_top_updates = ((steps_per_task // batch_size) * num_permutations) // topology_update_freq
            results_dict["prop_added_then_removed"] = torch.zeros(total_top_updates, **defaults)
        if extended_summaries:
            results_dict["loss_before_topology_update"] = []
            results_dict["loss_after_topology_update"] = []
            results_dict["avg_grad_before_topology_update"] = []
            results_dict["avg_grad_after_topology_update"] = []
            if use_ln:
                results_dict["change_in_average_activation_layer_1"] = []
                results_dict["change_in_average_activation_layer_2"] = []
                results_dict["change_in_average_activation_layer_3"] = []
                results_dict["change_in_std_activation_layer_1"] = []
                results_dict["change_in_std_activation_layer_2"] = []
                results_dict["change_in_std_activation_layer_3"] = []

    if extended_summaries:
        results_dict["average_gradient_magnitude_per_checkpoint"] = torch.zeros(total_ckpts, **defaults)
        results_dict["average_weight_magnitude_per_permutation"] = torch.zeros(num_permutations, **defaults)
        results_dict["proportion_dead_units_per_permutation"] = torch.zeros(num_permutations, **defaults)
        results_dict["stable_rank_per_permutation"] = torch.zeros(num_permutations, **defaults)
        if use_ln:
            results_dict["average_ln_weight_magnitude_per_checkpoint"] = torch.zeros(total_ckpts, **defaults)

    return results_dict
