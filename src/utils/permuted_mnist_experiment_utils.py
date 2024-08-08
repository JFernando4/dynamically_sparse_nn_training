# third party libraries
import torch
from torch.utils.data import DataLoader

# from src
from src.networks import ThreeHiddenLayerNetwork

@torch.no_grad()
def compute_average_weight_magnitude(net: ThreeHiddenLayerNetwork):
    """ computes the average weight magnitude of the network """

    weight_magnitude = 0.0
    total_weights = 0.0

    for p in net.parameters():
        if p.requires_grad:
            weight_magnitude += p.abs().sum()
            total_weights += p.numel()

    return weight_magnitude / total_weights


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
