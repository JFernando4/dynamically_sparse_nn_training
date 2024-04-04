import torch
from typing import Callable


def prune_and_grow_weights(weight: torch.Tensor,
                           prune_function: Callable[[torch.Tensor], None],
                           grow_function: Callable[[torch.Tensor], None]) -> tuple[torch.Tensor, int]:
    """ Prunes and grows weight in a weight matrix"""

    prune_function(weight)

    # get mask of pruned weights and compute number of pruned weights
    mask = torch.ones_like(weight, requires_grad=False)
    pruned_indices = torch.where(weight.flatten() == 0.0)[0]
    mask.view(-1)[pruned_indices] = 0.0
    num_pruned = len(pruned_indices)

    grow_function(weight)

    return mask, num_pruned


def setup_cbpw_weight_update_function(prune_name: str, grow_name: str, **kwargs) -> Callable[[torch.Tensor], tuple]:
    """ Sets up weight update function for CBP-w """
    prune_function_names = ["magnitude", "redo", "gf_redo", "gf", "hess_approx"]
    grow_function_names = ["pm_min", "kaiming_normal", "xavier_normal", "zero", "kaming_uniform", "xavier_uniform",
                           "fixed"]
    assert prune_name in prune_function_names and grow_name in grow_function_names
    assert "drop_factor" in kwargs.keys()

    if prune_name == "magnitude":
        prune_func = lambda w: magnitude_prune_weights(w, drop_factor=kwargs["drop_factor"])
    elif prune_name == "redo":
        prune_func = lambda w: redo_prune_weights(w, drop_factor=kwargs["drop_factor"])
    elif prune_name == "gf":
        prune_func = lambda w: gradient_flow_prune_weights(w, drop_factor=kwargs["drop_factor"])
    elif prune_name == "hess_approx":
        prune_func = lambda w: hessian_approx_prune_weights(w, drop_factor=kwargs["drop_factor"])

    if grow_name == "pm_min":
        grow_func = lambda w: pm_min_reinit_weights(w)
    elif "kaiming" in grow_name or "xavier" in grow_name:
        grow_func = lambda w: random_reinit_weights(w, reinit=grow_name)
    elif grow_name == "zero":
        grow_func = lambda w: None
    elif grow_name == "fixed":
        assert "reinit_val" in kwargs.keys()
        grow_func = lambda w: fixed_reinit_weights(w, kwargs["reinit_val"])

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


# ----- ----- ----- ----- Pruning Functions ----- ----- ----- ----- #
@torch.no_grad()
def redo_prune_weights(weight: torch.Tensor, drop_factor: float):
    """
    Prunes the weight that are smaller than (drop_factor * average_absolute_weight_value)
    """

    abs_weights = weight.abs().flatten()
    prune_threshold = drop_factor * abs_weights.mean()
    prune_indices = torch.where(abs_weights < prune_threshold)[0]
    weight.view(-1)[prune_indices] = 0.0


@torch.no_grad()
def magnitude_prune_weights(weight: torch.Tensor, drop_factor: int):
    """ Creates a mask by dropping the weights with the smallest magnitude """

    abs_weight = torch.abs(weight).flatten()
    drop_num = int(weight.numel() * drop_factor)
    indices = torch.argsort(abs_weight)
    weight.view(-1)[indices[:drop_num]] = 0.0


@torch.no_grad()
def gradient_flow_prune_weights(weight: torch.Tensor, drop_factor: int):
    """ Creates a mask by dropping the weights with the smallest gradient flow """

    gradient_flow = torch.abs(weight * weight.grad).flatten()
    drop_num = int(weight.numel() * drop_factor)
    indices = torch.argsort(gradient_flow)
    weight.view(-1)[indices[:drop_num]] = 0.0


@torch.no_grad()
def hessian_approx_prune_weights(weight: torch.Tensor, drop_factor: float):
    """
    Prunes using redo criteria but using gradient flow instead of magnitude pruning
    """

    hess_approx = torch.abs(weight.flatten().square() * weight.grad.flatten().square())
    drop_num = int(weight.numel() * drop_factor)
    indices = torch.argsort(hess_approx)
    weight.view(-1)[indices[:drop_num]] = 0.0


@torch.no_grad()
def threshold_prune_weights(weight: torch, drop_factor: float) -> None:
    """
    Prunes any weight whose absolute value is below the given drop_factor
    """
    abs_weight = weight.flatten().abs()
    indices = torch.where(abs_weight < drop_factor)[0]
    weight.view(-1)[indices] = 0.0


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
    weight.view(-1)[pruned_indices[len(pruned_indices) // 2:]] = min_abs_active
    weight.view(-1)[pruned_indices[:len(pruned_indices) // 2]] = -min_abs_active


@torch.no_grad()
def random_reinit_weights(weight: torch.Tensor, reinit) -> None:
    """
    Reinitializes entries in the weight matrix at the given indices using the specified reinit function

    Args:
        weight: torch.Tensor of weights
        reinit: name of reinitialization function. Should be in reinit_functions.key()
    """
    random_reinit_functions = {
        "kaiming_normal": torch.nn.init.kaiming_normal_,
        "kaiming_uniform_": torch.nn.init.kaiming_uniform,
        "xavier_normal": torch.nn.init.xavier_normal_,
        "xavier_uniform_": torch.nn.init.xavier_uniform_
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
