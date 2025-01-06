import torch
import numpy as np
from typing import Callable

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
    prune_function_names = ["magnitude", "gf", "efi", "mr", "gr", "er", "tgf", "tgr"]
    grow_function_names = ["kaiming_normal", "xavier_normal", "zero", "kaiming_uniform", "xavier_uniform", "fixed",
                           "clipped", "truncated", "median_clipped", "median_truncated", "25p_clipped", "25p_truncated",
                           "mean_truncated", "mean_clipped", "normal", "truncated_normal", "tx_uniform", "tx_normal",
                           "tk_uniform", "tk_normal"]
    assert prune_name in prune_function_names and grow_name in grow_function_names
    assert "drop_factor" in kwargs.keys()

    if prune_name == "magnitude":
        prune_func = lambda w: fixed_proportion_prune_weights(w, drop_factor=kwargs["drop_factor"], utility_name="magnitude")
    elif prune_name == "gf":    # gradient flow
        prune_func = lambda w: fixed_proportion_prune_weights(w, drop_factor=kwargs["drop_factor"], utility_name="gradient")
    elif prune_name == "tgf" or prune_name == "tgm":    # traced gradient flow or traced magnitude
        prune_func = lambda w: fixed_proportion_prune_weights(w, drop_factor=kwargs["drop_factor"], utility_name="gradient")
    elif prune_name == "efi":
        prune_func = lambda w: fixed_proportion_prune_weights(w, drop_factor=kwargs["drop_factor"], utility_name="efi")
    elif prune_name == "mr":    # magnitude redo
        prune_func = lambda w: redo_prune_weights(w, drop_factor=kwargs["drop_factor"], utility_name="magnitude")
    elif prune_name == "gr":    # gradient redo
        prune_func = lambda w: redo_prune_weights(w, drop_factor=kwargs["drop_factor"], utility_name="gradient")
    elif prune_name == "tgr" or prune_name == "tgm":   # traced gradient redo or traced magnitude redo
        prune_func = lambda w: redo_prune_weights(w, drop_factor=kwargs["drop_factor"], utility_name="trace")
    elif prune_name == "er":    # empirical fisher redo
        prune_func = lambda w: redo_prune_weights(w, drop_factor=kwargs["drop_factor"], utility_name="efi")

    activation = "relu" if "activation" not in kwargs else kwargs["activation"]
    fan_mode = "fan_in" if "fan_mode" not in kwargs else kwargs["fan_mode"]
    std = 0.01 if "std" not in kwargs else kwargs["std"]
    a = 0 if "a" not in kwargs else kwargs["a"]     # negative slope for leaky relu

    if "kaiming" in grow_name or "xavier" in grow_name:
        grow_func = lambda w, pi, ai: random_reinit_weights(w, pruned_indices=pi, active_indices=ai, reinit=grow_name, activation=activation, fan_mode=fan_mode, a=a)
    elif grow_name == "zero":
        grow_func = lambda w, pi, ai: fixed_reinit_weights(w, pruned_indices=pi, active_indices=ai, reinit_val=0.0)
    elif grow_name == "fixed":
        grow_func = lambda w, pi, ai: fixed_reinit_weights(w, pruned_indices=pi, active_indices=ai, reinit_val=kwargs["reinit_val"])
    elif grow_name == "clipped":
        grow_func = lambda w, pi, ai: clipped_reinit_weights(w, pruned_indices=pi, active_indices=ai, bound_method="min")
    elif grow_name == "median_clipped":
        grow_func = lambda w, pi, ai: clipped_reinit_weights(w, pruned_indices=pi, active_indices=ai, bound_method="median")
    elif grow_name == "25p_clipped":
        grow_func = lambda w, pi, ai: clipped_reinit_weights(w, pruned_indices=pi, active_indices=ai, bound_method="25p")
    elif grow_name == "mean_clipped":
        grow_func = lambda w, pi, ai: clipped_reinit_weights(w, pruned_indices=pi, active_indices=ai, bound_method="mean")
    elif grow_name == "truncated":
        grow_func = lambda w, pi, ai: bounded_kaiming_reinit_weights(w, pruned_indices=pi, active_indices=ai, bound_method="min")
    elif grow_name == "median_truncated":
        grow_func = lambda w, pi, ai: bounded_kaiming_reinit_weights(w, pruned_indices=pi, active_indices=ai, bound_method="median", activation=activation)
    elif grow_name == "25p_truncated":
        grow_func = lambda w, pi, ai: bounded_kaiming_reinit_weights(w, pruned_indices=pi, active_indices=ai, bound_method="25p")
    elif grow_name == "mean_truncated":
        grow_func = lambda w, pi, ai: bounded_kaiming_reinit_weights(w, pruned_indices=pi, active_indices=ai, bound_method="mean")
    elif grow_name == "tk_normal":      # truncated kaiming normal
        grow_func = lambda w, pi, ai: truncated_kaiming_reinit_weights(w, pi, ai, dist_type="normal", mode=fan_mode, activation=activation, a=a)
    elif grow_name == "tk_uniform":     # truncated kaiming uniform
        grow_func = lambda w, pi, ai: truncated_kaiming_reinit_weights(w, pi, ai, dist_type="uniform", mode=fan_mode, activation=activation, a=a)
    elif grow_name == "tx_normal":      # truncated xavier normal
        grow_func = lambda w, pi, ai: truncated_xavier_reinit_weights(w, pruned_indices=pi, active_indices=ai, dist_type="normal")
    elif grow_name == "tx_uniform":     # truncated xavier normal
        grow_func = lambda w, pi, ai: truncated_xavier_reinit_weights(w, pruned_indices=pi, active_indices=ai, dist_type="uniform")
    elif grow_name == "normal":
        grow_func = lambda w, pi, ai: normal_reinit_weights(w, pruned_indices=pi, active_indices=ai, std=std, truncated=False)
    elif grow_name == "truncated_normal":
        grow_func = lambda w, pi, ai: normal_reinit_weights(w, pruned_indices=pi, active_indices=ai, std=std, truncated=True)

    def temp_prune_and_grow_weights(w: torch.Tensor):
        return prune_and_grow_weights(w, prune_func, grow_func)

    return temp_prune_and_grow_weights


def update_weights(weight_dict: dict[str, tuple], reinitialization_rate: float = None) -> dict:
    """ Applies the corresponding update function to all the weights in the dictionary """
    summaries_dict = {}
    for k, v in weight_dict.items():
        reinit = 1 if reinitialization_rate is None else np.random.binomial(p=reinitialization_rate, n=1)
        if reinit:
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


def setup_cbpw_layer_norm_update_function(prune_name: str, drop_factor: float, exclude_layer_bias: bool = False
                                          ) -> Callable[[torch.nn.Module], None]:
    """ Sets up weight update function for CBP-w for layer or batch norm """
    prune_function_names = ["magnitude", "redo", "gf_redo", "gf"]
    assert prune_name in prune_function_names

    if prune_name == "magnitude":
        prune_func = lambda w: fixed_proportion_prune_weights(w, drop_factor=drop_factor, utility_name="magnitude")
    elif prune_name == "gf":
        prune_func = lambda w: fixed_proportion_prune_weights(w, drop_factor=drop_factor, utility_name="gradient")

    def temp_prune_and_grow_weights(w: torch.nn.Module):
        return update_norm_layer(w, prune_func, exclude_layer_bias)

    return temp_prune_and_grow_weights


# ----- ----- ----- ----- Pruning Functions ----- ----- ----- ----- #
@torch.no_grad()
def redo_prune_weights(weight: torch.Tensor, drop_factor: float, utility_name: str = "magnitude"):
    """
    Prunes the weight that are smaller than (drop_factor * average_absolute_weight_value)

    arguments:
        utility_name (str): "magnitude", "gradient", "efi" (empirical_fishe_information)
    """

    utility = compute_utility(weight, utility_name)

    prune_threshold = drop_factor * utility.mean()
    prune_indices = torch.where(utility < prune_threshold)[0]
    # print(f"{prune_indices.numel() = }")
    active_indices = torch.where(utility >= prune_threshold)[0]

    if utility_name == "trace":
        weight.utility_trace = None

    return prune_indices, active_indices


def fixed_proportion_prune_weights(weight: torch.Tensor, drop_factor: float, utility_name: str = "magnitude") \
        -> tuple[torch.Tensor, torch.Tensor]:
    """ Generates a tensor of indices to be pruned according to their utility """

    drop_num = compute_drop_num(weight.numel(), drop_factor)
    if drop_num == 0: return torch.empty(0), torch.arange(weight.numel())

    utility = compute_utility(weight, utility_name)

    indices = torch.argsort(utility)
    pruned_indices = indices[:drop_num]
    active_indices = indices[drop_num:]
    return pruned_indices, active_indices


@torch.no_grad()
def compute_utility(weight: torch.Tensor, utility_name: str = "magnitude") -> torch.Tensor:
    """
    Computes the utility of the weights in the given tensor

    arguments:
        weight (torch.Tensor): tensor with weights
        utility_name (str): "magnitude", "gradient", "efi" (empirical_fishe_information)

    returns:
        Tensor of utilities
    """

    if utility_name == "magnitude":
        utility = weight.abs().flatten()
    elif utility_name == "gradient":
        utility = torch.abs(weight * weight.grad).flatten()
    elif utility_name == "efi":
        assert hasattr(weight, "empirical_fisher")
        utility = weight.empirical_fisher.flatten()
    elif utility_name == "trace":
        assert hasattr(weight, "utility_trace")
        utility = weight.utility_trace
    else:
        raise ValueError(f"{utility_name} is not a valid utility.")

    return utility


def compute_drop_num(num_weights: int, drop_factor: float) -> int:
    """ Computes the number of weights dropped """
    fraction_to_prune = num_weights * drop_factor
    drop_num = int(fraction_to_prune) + np.random.binomial(n=1, p=fraction_to_prune % 1, size=None)
    return drop_num


@torch.no_grad()
def compute_trace_utility(weight:torch.Tensor, decay_rate: float = 0.99, utility_name: str = "magnitude"):
    """ Computes a trace of the utility function and stores it in a new attribute in the weight called utility_trace """

    if utility_name == "magnitude":
        utility = torch.abs(weight).flatten()
    elif utility_name == "gradient":
        utility = torch.abs(weight * weight.grad).flatten()
    else:
        raise ValueError(f"{utility_name} is not a valid utility function.")

    if hasattr(weight, "utility_trace"):
        if weight.utility_trace is None:
            weight.utility_trace = utility
        else:
            weight.utility_trace *= decay_rate
            weight.utility_trace += (1 - decay_rate) * utility
    else:
        weight.utility_trace = utility


# ----- ----- ----- ----- Growing Functions ----- ----- ----- ----- #
@torch.no_grad()
def clipped_reinit_weights(weight: torch.Tensor,  pruned_indices: torch.Tensor, active_indices: torch.Tensor,
                           activation: str = "relu", bound_method: str = "min") -> None:
    """
    Reinitializes entries in teh wegith matrix at the given indices using clipped kaiming reinitialization
    """

    clip_value = get_bounding_value(weight, active_indices, bound_method)
    gain = torch.nn.init.calculate_gain(activation)
    fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(weight)
    std = gain / np.sqrt(fan_in)                                                # kaiming normal standard deviation
    # print(f"{clip_value = }, {std = }")
    new_weights = torch.randn(size=pruned_indices.size(), device=weight.device) * std
    clipped_new_weights = torch.clip(new_weights, -clip_value, clip_value)
    weight.view(-1)[pruned_indices] = clipped_new_weights


@torch.no_grad()
def bounded_kaiming_reinit_weights(weight: torch.Tensor, pruned_indices: torch.Tensor, active_indices: torch.Tensor,
                                   activation: str = "relu", bound_method: str = "min") -> None:
    """
    Reinitializes entries in the weight tensor at the given indices using truncated or clipped kaiming reinitialization
    """

    truncation_value = get_bounding_value(weight, active_indices, bound_method)
    gain = torch.nn.init.calculate_gain(activation)
    fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(weight)
    std = gain / np.sqrt(fan_in)                                                # kaiming normal standard deviation
    # print(f"{truncation_value = }, {std = }")
    new_weights = torch.zeros(size=pruned_indices.size(), dtype=weight.dtype, device=weight.device)
    torch.nn.init.trunc_normal_(new_weights, mean=0, std=std, a=-truncation_value, b=truncation_value)

    weight.view(-1)[pruned_indices] = new_weights


@torch.no_grad()
def truncated_kaiming_reinit_weights(weight: torch.Tensor, pruned_indices: torch.Tensor, active_indices: torch.Tensor,
                             activation: str = "relu", dist_type: str = "normal", mode: str = "fan_in",
                             bound_method: str = "median", a: float = 0.0) -> None:
    """
    Reinitializes entries in teh wegith matrix at the given indices using clipped kaiming reinitialization

    Parameters:
        activation: should be in ["relu", "leaky_relu"]
        dist_type: should be in ["normal", "uniform"]
        mode: should be in ["fan_in", "fan_out"]
        bound_method: should be in ["median", "min", "mean", "25p"]
        a: the negative slope for leaky relu
    """

    truncation_value = get_bounding_value(weight, active_indices, bound_method)

    gain = torch.nn.init.calculate_gain(activation, param=a)
    fan = torch.nn.init._calculate_correct_fan(weight, mode)

    new_weights = torch.zeros(size=pruned_indices.size(), dtype=weight.dtype, device=weight.device)
    if dist_type == "normal":
        kaiming_normal_std = gain / np.sqrt(fan)
        torch.nn.init.trunc_normal_(new_weights, mean=0, std=kaiming_normal_std, a=-truncation_value, b=truncation_value)
    elif dist_type == "uniform":
        kaiming_uniform_bound = gain * np.sqrt(3) / np.sqrt(fan)
        bound = min(kaiming_uniform_bound, truncation_value)
        torch.nn.init.uniform_(new_weights, -bound, bound)
    else:
        raise ValueError(f"{dist_type} is not a valid dist_type.")

    weight.view(-1)[pruned_indices] = new_weights


@torch.no_grad()
def truncated_xavier_reinit_weights(weight: torch.Tensor, pruned_indices: torch.Tensor, active_indices: torch.Tensor,
                                    dist_type: str = "normal", bound_method: str = "median") -> None:
    """
    Reinitializes entries in teh wegith matrix at the given indices using clipped kaiming reinitialization

    Parameters:
        dist_type: should be in ["normal", "uniform"]
        bound_method: should be in ["median", "min", "mean", "25p"]
    """

    truncation_value = get_bounding_value(weight, active_indices, bound_method)

    fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(weight)

    new_weights = torch.zeros(size=pruned_indices.size(), dtype=weight.dtype, device=weight.device)
    if dist_type == "normal":
        xavier_normalstd = np.sqrt(2) / np.sqrt(fan_in + fan_out)
        torch.nn.init.trunc_normal_(new_weights, mean=0, std=xavier_normalstd, a=-truncation_value, b=truncation_value)
    elif dist_type == "uniform":
        xavier_uniform_bound = np.sqrt(6) / np.sqrt(fan_in + fan_out)
        bound = min(truncation_value, xavier_uniform_bound)
        torch.nn.init.uniform_(new_weights, -bound, bound)
    else:
        raise ValueError(f"{dist_type} is not a valid dist_type.")

    weight.view(-1)[pruned_indices] = new_weights


@torch.no_grad()
def normal_reinit_weights(weight: torch.Tensor, pruned_indices: torch.Tensor, active_indices: torch.Tensor,
                          std: float = 0.01, truncated: bool = False) -> None:
    """
    Reinitializes weights using a normal distribution with the given standard deviation. If truncated is true, the
    weights are reinitialized with a truncated normal distribution with truncation equal to the median of the absolute
    value of the active weights
    """

    new_weights = torch.zeros(size=pruned_indices.size(), dtype=weight.dtype, device=weight.device)
    if truncated:
        truncation_value = get_bounding_value(weight, active_indices, bound_method="median")
        torch.nn.init.trunc_normal_(new_weights, mean=0, std=std, a=-truncation_value, b=truncation_value)
    else:
        torch.nn.init.normal_(new_weights, mean=0, std=std)

    weight.view(-1)[pruned_indices] = new_weights


@torch.no_grad()
def get_bounding_value(weight: torch.Tensor, active_indices: torch.Tensor, bound_method: str) -> float:
    """
    Returns the value that bounds above and below the distribution of new weights
    """
    abs_weights = weight.flatten().abs()[active_indices]
    if bound_method == "median":
        return float(abs_weights.median())
    elif bound_method == "min":
        return float(abs_weights.min())
    elif bound_method == "25p":
        return float(torch.quantile(abs_weights, 0.25))
    elif bound_method == "mean":
        return float(abs_weights.mean())
    else:
        raise ValueError(f"{bound_method} is not a valid bound method!")


@torch.no_grad()
def random_reinit_weights(weight: torch.Tensor, pruned_indices: torch.Tensor, active_indices: torch.Tensor, reinit,
                          activation: str = "relu", fan_mode: str = "fan_in", a: float = 0.0) -> None:
    """
    Reinitializes entries in the weight matrix at the given indices using the specified reinit function

    Args:
        weight: torch.Tensor of weights
        reinit: name of reinitialization function. Should be in reinit_functions.key()
    """
    random_reinit_functions = {
        "kaiming_normal": lambda m: torch.nn.init.kaiming_normal_(m, nonlinearity=activation, mode=fan_mode, a=a),
        "kaiming_uniform": lambda m: torch.nn.init.kaiming_uniform_(m, nonlinearity=activation, mode=fan_mode, a=a),
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
