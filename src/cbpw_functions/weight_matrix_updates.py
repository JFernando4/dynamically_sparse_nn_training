import torch
import numpy as np
from typing import Callable
from scipy.special import erfinv


def prune_and_grow_weights(weight: torch.Tensor,
                           prune_function: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
                           grow_function: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], None]) -> tuple[torch.Tensor, int]:
    """ Prunes and grows weight in a weight matrix"""

    pruned_indices, active_indices = prune_function(weight)

    # get mask of pruned weights and compute number of pruned weights
    mask = torch.ones_like(weight, requires_grad=False)
    if len(pruned_indices) == 0:
        return mask, 0
    mask.view(-1)[pruned_indices] = 0.0
    num_pruned = len(pruned_indices)

    grow_function(weight, pruned_indices, active_indices)

    return mask, num_pruned


def setup_cbpw_weight_update_function(prune_name: str, grow_name: str, **kwargs) -> Callable[[torch.Tensor], tuple]:
    """ Sets up weight update function for CBP-w """
    prune_function_names = ["magnitude", "gf"]
    grow_function_names = ["kaiming_normal", "xavier_normal", "zero", "kaming_uniform", "xavier_uniform",
                           "fixed", "clipped", "mad", "truncated"]
    assert prune_name in prune_function_names and grow_name in grow_function_names
    assert "drop_factor" in kwargs.keys()

    as_rate = False if "as_rate" not in kwargs.keys() else kwargs["as_rate"]
    if prune_name == "magnitude":
        prune_func = lambda w: magnitude_prune_weights(w, drop_factor=kwargs["drop_factor"], as_rate=as_rate)
    elif prune_name == "gf":
        prune_func = lambda w: gradient_flow_prune_weights(w, drop_factor=kwargs["drop_factor"], as_rate=as_rate)

    if "kaiming" in grow_name or "xavier" in grow_name:
        grow_func = lambda w, pi, ai: random_reinit_weights(w, pruned_indices=pi, active_indices=ai, reinit=grow_name)
    elif grow_name == "zero":
        grow_func = lambda w, pi, ai: fixed_reinit_weights(w, pruned_indices=pi, active_indices=ai, reinit_val=0.0)
    elif grow_name == "fixed":
        assert "reinit_val" in kwargs.keys()
        grow_func = lambda w, pi, ai: fixed_reinit_weights(w, pruned_indices=pi, active_indices=ai, reinit_val=kwargs["reinit_val"])
    elif grow_name == "clipped":
        grow_func = lambda w, pi, ai: clipped_reinit_weights(w, pruned_indices=pi, active_indices=ai)
    elif grow_name == "mad":
        grow_func = lambda w, pi, ai: magnitude_adjusted_uniform_reinit_weights(w, pruned_indices=pi, active_indices=ai)
    elif grow_name == "truncated":
        grow_func = lambda w, pi, ai: truncated_normal_reinit_weights(w, pruned_indices=pi, active_indices=ai)

    def temp_prune_and_grow_weights(w: torch.Tensor):
        return prune_and_grow_weights(w, prune_func, grow_func)

    return temp_prune_and_grow_weights


def update_weights(weight_dict: dict[str, tuple]) -> dict:
    """ Applies the corresponding update function to all the weights in the dictionary """
    summaries_dict = {}
    for k, v in weight_dict.items():
        temp_weight, temp_update_function = v
        summaries_dict[k] = temp_update_function(temp_weight)
    return summaries_dict


@torch.no_grad()
def update_norm_layer(norm_layer: torch.nn.Module,
                      prune_function: Callable[[torch.Tensor], None],
                      exclude_bn_bias: bool = False) -> None:
    assert isinstance(norm_layer, (torch.nn.LayerNorm, torch.nn.BatchNorm2d))

    prune_function(norm_layer.weight)
    pruned_indices = torch.where(norm_layer.weight.flatten() == 0.0)[0]
    norm_layer.weight[pruned_indices] = 1.0
    if not exclude_bn_bias:
        norm_layer.bias[pruned_indices] = 0.0


def setup_cbpw_layer_norm_update_function(prune_name: str, drop_factor: float, exclude_layer_bias: bool = False,
                                          as_rate: bool = False) -> Callable[[torch.nn.Module], None]:
    """ Sets up weight update function for CBP-w for layer or batch norm """
    prune_function_names = ["magnitude", "redo", "gf_redo", "gf"]
    assert prune_name in prune_function_names

    if prune_name == "magnitude":
        prune_func = lambda w: magnitude_prune_weights(w, drop_factor=drop_factor)
    elif prune_name == "redo":
        prune_func = lambda w: magnitude_redo_prune_weights(w, drop_factor=drop_factor)
    elif prune_name == "gf":
        prune_func = lambda w: gradient_flow_prune_weights(w, drop_factor=drop_factor)

    def temp_prune_and_grow_weights(w: torch.nn.Module):
        return update_norm_layer(w, prune_func, exclude_layer_bias)

    return temp_prune_and_grow_weights


# ----- ----- ----- ----- Pruning Functions ----- ----- ----- ----- #
@torch.no_grad()
def magnitude_redo_prune_weights(weight: torch.Tensor, drop_factor: float):
    """
    Prunes the weight that are smaller than (drop_factor * average_absolute_weight_value)
    """

    abs_weights = weight.abs().flatten()
    prune_threshold = drop_factor * abs_weights.mean()
    prune_indices = torch.where(abs_weights < prune_threshold)[0]
    if len(prune_indices) > 0:                      # prune according to redo
        weight.view(-1)[prune_indices] = 0.0
    else:                                           # prune one weight according to magnitude pruning
        magnitude_prune_weights(weight, drop_factor)


@torch.no_grad()
def magnitude_prune_weights(weight: torch.Tensor, drop_factor: float, as_rate: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """ Creates a mask by dropping the weights with the smallest magnitude """

    drop_num = compute_drop_num(weight.numel(), drop_factor, as_rate)
    if drop_num == 0: return torch.empty(0), torch.empty(0)

    abs_weight = torch.abs(weight).flatten()
    indices = torch.argsort(abs_weight)
    pruned_indices = indices[:drop_num]
    active_indices = indices[drop_num:]
    return pruned_indices, active_indices


@torch.no_grad()
def gradient_flow_prune_weights(weight: torch.Tensor, drop_factor: float, as_rate: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """ Creates a mask by dropping the weights with the smallest gradient flow """

    drop_num = compute_drop_num(weight.numel(), drop_factor, as_rate)
    if drop_num == 0: return torch.empty(0), torch.empty(0)

    gradient_flow = torch.abs(weight * weight.grad).flatten()
    indices = torch.argsort(gradient_flow)
    pruned_indices = indices[:drop_num]
    active_indices = indices[drop_num:]
    return pruned_indices, active_indices


def compute_drop_num(num_weights: int, drop_factor: float, as_rate: bool = False) -> int:
    """ Computes the number of weights dropped """
    fraction_to_prune = num_weights * drop_factor
    if as_rate:
        drop_num = int(fraction_to_prune) + np.random.binomial(n=1, p=fraction_to_prune % 1, size=None)
    else:
        drop_num = max(int(fraction_to_prune), 1)   # drop at least one weight
    return drop_num

# ----- ----- ----- ----- Growing Functions ----- ----- ----- ----- #
@torch.no_grad()
def clipped_reinit_weights(weight: torch.Tensor,  pruned_indices: torch.Tensor, active_indices: torch.Tensor,
                           activation: str = "relu", clip_to_median: bool = False) -> None:
    """
    Reinitializes entries in teh wegith matrix at the given indices using clipped kaiming reinitialization
    """

    if clip_to_median:
        clip_value = weight.flatten().abs()[active_indices].median()
    else:
        clip_value = weight.flatten().abs()[active_indices].min()
    gain = torch.nn.init.calculate_gain(activation)
    fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(weight)
    std = gain / np.sqrt(fan_in)                                                # kaiming normal standard deviation

    new_weights = torch.randn(size=pruned_indices.size()) * std
    clipped_new_weights = torch.clip(new_weights, -clip_value, clip_value)
    weight.view(-1)[pruned_indices] = clipped_new_weights


@torch.no_grad()
def truncated_normal_reinit_weights(weight: torch.Tensor, pruned_indices: torch.Tensor, active_indices: torch.Tensor,
                                    activation: str = "relu", truncate_to_median: bool = False) -> None:
    """
    Reinitializes entries in teh wegith matrix at the given indices using clipped kaiming reinitialization
    """

    if truncate_to_median:
        clip_value = weight.flatten().abs()[active_indices].median()
    else:
        clip_value = weight.flatten().abs()[active_indices].min()
    gain = torch.nn.init.calculate_gain(activation)
    fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(weight)
    std = gain / np.sqrt(fan_in)                                                # kaiming normal standard deviation

    new_weights = torch.zeros(size=pruned_indices.size(), dtype=weight.dtype, device=weight.device)
    torch.nn.init.trunc_normal_(new_weights, mean=0, std=std, a=-clip_value, b=clip_value)

    weight.view(-1)[pruned_indices] = new_weights


@torch.no_grad()
def magnitude_adjusted_uniform_reinit_weights(weight: torch,  pruned_indices: torch.Tensor, active_indices: torch.Tensor):
    """
    Reinitializes entries in the weight matrix at the given indices using U(-median_active, median_active)
    This way, the new weights will have an average magnitude of mean_active
    """

    mean_pruned_weights = weight.flatten().abs()[pruned_indices].mean()
    new_weights = torch.randn(size=pruned_indices.size()) * (np.sqrt(np.pi/2) * mean_pruned_weights)
    # This: (r1 - r2) * torch.rand(a, b) + r2, gives samples from a uniform distribution in interval [r1, r2]
    # new_weights = - (-2 * torch.rand(size=pruned_indices.size()) + 1) * 2 * mean_active
    # print(f"Standard Deviation = {np.sqrt(np.pi/2) * mean_pruned_weights}, {mean_pruned_weights = }")
    weight.view(-1)[pruned_indices] = new_weights


@torch.no_grad()
def random_reinit_weights(weight: torch.Tensor, pruned_indices: torch.Tensor, active_indices: torch.Tensor, reinit) -> None:
    """
    Reinitializes entries in the weight matrix at the given indices using the specified reinit function

    Args:
        weight: torch.Tensor of weights
        reinit: name of reinitialization function. Should be in reinit_functions.key()
    """
    random_reinit_functions = {
        "kaiming_normal": lambda m: torch.nn.init.kaiming_normal_(m, nonlinearity="relu"),
        "kaiming_uniform": lambda m: torch.nn.init.kaiming_uniform_(m, nonlinearity="relu"),
        "xavier_normal": torch.nn.init.xavier_normal_,
        "xavier_uniform": torch.nn.init.xavier_uniform_
    }
    assert reinit in random_reinit_functions.keys()

    temp_weights = torch.empty_like(weight)
    random_reinit_functions[reinit](temp_weights)
    weight.view(-1)[pruned_indices] = temp_weights.view(-1)[pruned_indices]


@torch.no_grad()
def fixed_reinit_weights(weight: torch.Tensor, pruned_indices: torch.Tensor, active_indices: torch.Tensor, reinit_val: float) -> None:
    """ Reinitializes weights toa fixed value """
    weight.view(-1)[pruned_indices] = reinit_val
