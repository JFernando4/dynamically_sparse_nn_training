import torch
import numpy as np
from typing import Callable


def prune_and_grow_weights(weight: torch.Tensor,
                           prune_function: Callable[[torch.Tensor], None],
                           grow_function: Callable[[torch.Tensor], None]) -> tuple[torch.Tensor, int]:
    """ Prunes and grows weight in a weight matrix"""

    was_pruned = prune_function(weight)

    # get mask of pruned weights and compute number of pruned weights
    mask = torch.ones_like(weight, requires_grad=False)
    if not was_pruned:
        return mask, 0
    pruned_indices = torch.where(weight.flatten() == 0.0)[0]
    mask.view(-1)[pruned_indices] = 0.0
    num_pruned = len(pruned_indices)

    grow_function(weight)

    return mask, num_pruned


def setup_cbpw_weight_update_function(prune_name: str, grow_name: str, **kwargs) -> Callable[[torch.Tensor], tuple]:
    """ Sets up weight update function for CBP-w """
    prune_function_names = ["magnitude", "redo", "gf_redo", "gf", "hess_approx"]
    grow_function_names = ["pm_min", "kaiming_normal", "xavier_normal", "zero", "kaming_uniform", "xavier_uniform",
                           "fixed", "fixed_with_noise", "clipped", "cstd"]
    assert prune_name in prune_function_names and grow_name in grow_function_names
    assert "drop_factor" in kwargs.keys()

    as_rate = False if "as_rate" not in kwargs.keys() else kwargs["as_rate"]
    if prune_name == "magnitude":
        prune_func = lambda w: magnitude_prune_weights(w, drop_factor=kwargs["drop_factor"], as_rate=as_rate)
    elif prune_name == "redo":
        prune_func = lambda w: redo_prune_weights(w, drop_factor=kwargs["drop_factor"])
    elif prune_name == "gf":
        prune_func = lambda w: gradient_flow_prune_weights(w, drop_factor=kwargs["drop_factor"], as_rate=as_rate)
    elif prune_name == "hess_approx":
        mb_size = 1.0 if "mb_size" not in kwargs.keys() else kwargs["mb_size"]
        prune_func = lambda w: hessian_approx_prune_weights(w, drop_factor=kwargs["drop_factor"], mb_size=mb_size, as_rate=as_rate)

    if grow_name == "pm_min":
        grow_func = lambda w: pm_min_reinit_weights(w)
    elif "kaiming" in grow_name or "xavier" in grow_name:
        grow_func = lambda w: random_reinit_weights(w, reinit=grow_name)
    elif grow_name == "zero":
        grow_func = lambda w: None
    elif grow_name == "fixed":
        assert "reinit_val" in kwargs.keys()
        grow_func = lambda w: fixed_reinit_weights(w, kwargs["reinit_val"])
    elif grow_name == "fixed_with_noise":
        assert "reinit_val" in kwargs.keys()
        assert "noise_std" in kwargs.keys()
        grow_func = lambda w: fixed_reinit_weights_with_noise(w, kwargs["reinit_val"], kwargs["noise_std"])
    elif grow_name == "clipped":
        assert "activation" in kwargs.keys()
        grow_func = lambda w: clipped_reinit_weights(w, activation=kwargs["activation"])
    elif grow_name == "cstd":    # clipped std
        assert "activation"
        grow_func = lambda w: clipped_std_reinit_weights(w, activation=kwargs["activation"])

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
    prune_function_names = ["magnitude", "redo", "gf_redo", "gf", "hess_approx"]
    assert prune_name in prune_function_names

    if prune_name == "magnitude":
        prune_func = lambda w: magnitude_prune_weights(w, drop_factor=drop_factor)
    elif prune_name == "redo":
        prune_func = lambda w: redo_prune_weights(w, drop_factor=drop_factor)
    elif prune_name == "gf":
        prune_func = lambda w: gradient_flow_prune_weights(w, drop_factor=drop_factor)
    elif prune_name == "hess_approx":
        prune_func = lambda w: hessian_approx_prune_weights(w, drop_factor=drop_factor, as_rate=as_rate)

    def temp_prune_and_grow_weights(w: torch.nn.Module):
        return update_norm_layer(w, prune_func, exclude_layer_bias)

    return temp_prune_and_grow_weights


# ----- ----- ----- ----- Pruning Functions ----- ----- ----- ----- #
@torch.no_grad()
def redo_prune_weights(weight: torch.Tensor, drop_factor: float):
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
def magnitude_prune_weights(weight: torch.Tensor, drop_factor: float, as_rate: bool = False) -> bool:
    """ Creates a mask by dropping the weights with the smallest magnitude """

    drop_num = compute_drop_num(weight.numel(), drop_factor, as_rate)
    if drop_num == 0: return False

    abs_weight = torch.abs(weight).flatten()
    indices = torch.argsort(abs_weight)
    weight.view(-1)[indices[:drop_num]] = 0.0
    return True


@torch.no_grad()
def gradient_flow_prune_weights(weight: torch.Tensor, drop_factor: float, as_rate: bool = False) -> bool:
    """ Creates a mask by dropping the weights with the smallest gradient flow """

    drop_num = compute_drop_num(weight.numel(), drop_factor, as_rate)
    if drop_num == 0: return False

    gradient_flow = torch.abs(weight * weight.grad).flatten()
    indices = torch.argsort(gradient_flow)
    weight.view(-1)[indices[:drop_num]] = 0.0
    return True


@torch.no_grad()
def hessian_approx_prune_weights(weight: torch.Tensor, drop_factor: float, mb_size: float = 1.0, as_rate: bool = False):
    """
    Prunes using redo criteria but using gradient flow instead of magnitude pruning
    """

    drop_num = compute_drop_num(weight.numel(), drop_factor, as_rate)
    if drop_num == 0: return False

    hess_approx = torch.abs(weight.flatten().square() * (weight.grad.flatten() * mb_size).square())
    indices = torch.argsort(hess_approx)
    weight.view(-1)[indices[:drop_num]] = 0.0
    return True


@torch.no_grad()
def threshold_prune_weights(weight: torch, drop_factor: float) -> None:
    """
    Prunes any weight whose absolute value is below the given drop_factor
    """
    abs_weight = weight.flatten().abs()
    indices = torch.where(abs_weight < drop_factor)[0]
    weight.view(-1)[indices] = 0.0


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
def pm_min_reinit_weights(weight: torch.Tensor) -> None:
    """

    """
    pruned_indices = torch.where(weight.flatten() == 0.0)[0]
    active_indices = torch.where(weight.flatten() != 0.0)[0]

    if len(active_indices) == 0:
        return

    min_abs_active = weight.flatten().abs()[active_indices].min()
    random_sign = -1.0 if torch.rand(1) > 0.5 else 1.0
    weight.view(-1)[pruned_indices[len(pruned_indices) // 2:]] = random_sign * min_abs_active
    weight.view(-1)[pruned_indices[:len(pruned_indices) // 2]] = - random_sign * min_abs_active


@torch.no_grad()
def clipped_reinit_weights(weight: torch.Tensor, activation: str = "relu") -> None:
    """
    Reinitializes entries in teh wegith matrix at the given indices using clipped kaiming reinitialization
    """
    pruned_indices = torch.where(weight.flatten() == 0.0)[0]
    active_indices = torch.where(weight.flatten() != 0.0)[0]

    if len(active_indices) == 0:
        return

    min_abs_active = weight.flatten().abs()[active_indices].min()
    gain = torch.nn.init.calculate_gain(activation)
    fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(weight)
    std = gain / np.sqrt(fan_in)  # kaiming normal standard deviation

    new_weights = torch.randn(size=pruned_indices.size()) * std
    clipped_new_weights = torch.clip(new_weights, -min_abs_active, min_abs_active)
    weight.view(-1)[pruned_indices] = clipped_new_weights


@torch.no_grad()
def clipped_std_reinit_weights(weight: torch.Tensor, activation: str = "relu") -> None:
    """
    Reinitializes entries in teh wegith matrix at the given indices using kaiming reinitialization with clipped
    standard deviation, i.e., Normal(0, min(min_active_weight, kaiming_std))
    """
    pruned_indices = torch.where(weight.flatten() == 0.0)[0]
    active_indices = torch.where(weight.flatten() != 0.0)[0]

    if len(active_indices) == 0:
        return

    min_abs_active = weight.flatten().abs()[active_indices].min()
    gain = torch.nn.init.calculate_gain(activation)
    fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(weight)
    std = gain / np.sqrt(fan_in)  # kaiming normal standard deviation

    new_weights = torch.randn(size=pruned_indices.size()) * min(std, min_abs_active)
    weight.view(-1)[pruned_indices] = new_weights


@torch.no_grad()
def random_reinit_weights(weight: torch.Tensor, reinit) -> None:
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
    pruned_indices = torch.where(weight.flatten() == 0.0)[0]
    random_reinit_functions[reinit](temp_weights)
    weight.view(-1)[pruned_indices] = temp_weights.view(-1)[pruned_indices]


@torch.no_grad()
def fixed_reinit_weights(weight: torch.Tensor, reinit_val: float) -> None:
    """ Reinitializes weights toa fixed value """
    pruned_indices = torch.where(weight.flatten() == 0.0)[0]
    weight.view(-1)[pruned_indices] = reinit_val

@torch.no_grad()
def fixed_reinit_weights_with_noise(weight: torch.Tensor, reinit_val: float, noise_std: float) -> None:
    """ Reinitializes weights toa fixed value """
    pruned_indices = torch.where(weight.flatten() == 0.0)[0]
    weight.view(-1)[pruned_indices] = reinit_val + torch.randn_like(weight.view(-1)[pruned_indices]) * noise_std