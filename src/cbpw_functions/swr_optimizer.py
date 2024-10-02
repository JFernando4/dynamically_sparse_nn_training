"""
Modification of pytorch's SGD optimizer that applies l1 and l2 regularization plus connection-level continual backprop
"""
import numpy as np
import torch
from math import modf


def get_init_parameters(net: torch.nn.Module, initialization_type: str, activation: str = "relu",
                        scaling: float = 1.0) -> tuple[list[float], list[float]]:
    """ Returns the parameters of the distribution used to initialize each parameter in the network """
    means = get_means(net)
    stds = get_stds(net, initialization_type, activation, scaling)
    return means, stds


def get_means(net: torch.nn.Module) -> list[float]:
    """ Returns the mean of the distributions used to initialize each parameter in a network """
    means = []

    for n, p in net.named_parameters():
        is_weight = "weight" in n
        is_layer_or_batch_norm = ((".ln_1." in n) or (".ln_2." in n) or (".ln." in n) or ("bn1." in n) or ("bn2." in n)
                                  or ("downsample.1." in n) or ("ln_1." in n) or ("ln_2." in n) or ("ln_3." in n))

        if is_weight and is_layer_or_batch_norm:
            means.append(1.0)
        else:
            means.append(0.0)
    return means


def get_stds(net: torch.nn.Module, initialization_type: str, activation: str = "relu", scaling: float = 1.0) -> list[float]:
    """ Returns the standard deviation of the distribution used to initialize each parameter in a network """

    stds = []
    gain = torch.nn.init.calculate_gain(activation)

    for p in net.parameters():
        if not p.requires_grad:
            continue

        if len(p.shape) == 1:   # weight vector of batch or layer norm or bias term, both use constant initialization
            stds.append(0.0)
            continue

        fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(p)
        if initialization_type == "kaiming_normal":
            stds.append((gain / np.sqrt(fan_in)) * scaling)
        elif initialization_type == "xavier_normal":
            stds.append(gain * np.sqrt(2 / (fan_in + fan_out)) * scaling)
        else:  # Default to zero initialization
            stds.append(0.0)

    return stds


@torch.no_grad()
def compute_weight_magnitude(weight: torch.Tensor) -> torch.Tensor:
    """
    Computes weight magnitude utility: | w_i | for each w_i in weight tensor
    returns flat tensor with utilities
    """
    return torch.abs(weight).flatten()


@torch.no_grad()
def compute_gradient_flow(weight: torch.Tensor) -> torch.Tensor:
    """
    Computes gradient flow / gradient magnitude / saliency utility: | w_i g_i | for each w_i in weight tensor
    returns flat tensor with utilities
    """
    return torch.abs(weight * weight.grad).flatten()


@torch.no_grad()
def prune_number_of_weights(weight: torch.Tensor, drop_num: int, mature_weights_utility: torch.Tensor,
                            mature_weights_indices:torch.Tensor) -> torch.Tensor:
    """
    Sets the weights above a mature threshold and with low utility to zero
    returns indices of the pruned weights
    """
    sorted_utility_indices = torch.argsort(mature_weights_utility)
    pruned_indices = mature_weights_indices[sorted_utility_indices][:drop_num]
    weight.view(-1)[pruned_indices] = 0.0
    return pruned_indices


@torch.no_grad()
def reinitialize_weights_with_noise(weight: torch.Tensor, pruned_indices: torch.Tensor, center: float = 0.0,
                                    std: float = 0.0):
    """ Reinitializes the entries of the given weight tensor at the given indices to center + Normal(center, std) """

    if center == 0.0 and std == 0.0:
        return

    new_value = center + torch.zeros(size=pruned_indices.size())
    if std > 0.0:
        new_value = center + torch.randn(size=pruned_indices.size()) * std
    weight.view(-1)[pruned_indices] = new_value


class SelectiveWeightReinitializationSGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, weight_decay=0.0, l1_reg_factor=0.0, momentum=0.0,
                 replacement_rate: float = 0.0, maturity_threshold: int = 0, utility: str = None,
                 new_params_mean: list[float] = None, new_params_std: list[float] = None, beta_utility: float = 0.0):

        params = list(params)
        # Check that arguments hae the right values.
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if l1_reg_factor < 0.0:
            raise ValueError(f"Invalid l1_reg_factor value: {l1_reg_factor}")
        if replacement_rate < 0.0 or replacement_rate >= 1.0:
            raise ValueError(f"Invalid replacement_rate value: {replacement_rate}")
        if utility not in ["none", "magnitude", "gf"]:  # magnitude = weight magnitude, gm = gradient magnitude
            raise ValueError(f"Invalid pruning_function value: {utility}. Choose from: ['none', 'magnitude', 'gf']")
        if new_params_mean is not None:
            assert isinstance(new_params_mean, list)
            if not all([isinstance(temp_mean,float) for temp_mean in new_params_mean]):
                raise ValueError("The means of the new weights should be floats")
        if new_params_std is not None:
            assert isinstance(new_params_std, list)
            if not all([isinstance(temp_std,float) for temp_std in new_params_std]):
                raise ValueError("The standard deviations of the new weights should be floats.")
        if len(params) != len(new_params_mean) or len(params) != len(new_params_std):
            raise ValueError("The number of parameter tensors should be the same as number of entries in "
                             "new_params_mean and new_params_std.")

        defaults = dict(lr=lr, weight_decay=weight_decay, l1_reg_factor=l1_reg_factor, momentum=momentum,
                        replacement_rate=replacement_rate, maturity_threshold=maturity_threshold,
                        new_weights_mean=new_params_mean, new_weights_std=new_params_std, prune_method=utility,
                        beta_utility=beta_utility)

        self.utility_function = None
        if utility == "magnitude":
            self.utility_function = compute_weight_magnitude
        elif utility == "gf":
            self.utility_function = compute_gradient_flow

        super(SelectiveWeightReinitializationSGD, self).__init__(params, defaults)

        for p in params:
            if p.requires_grad:
                self.state[p]["momentum_buffer"] = torch.zeros_like(p, requires_grad=False)
                self.state[p]["utility_trace"] = torch.zeros_like(p, requires_grad=False).view(-1)
                self.state[p]["age"] = torch.zeros_like(p, requires_grad=False, dtype=torch.int32).view(-1)
                self.state[p]["num_reinit_per_step"] = p.numel() * replacement_rate
                self.state[p]["current_num_reinit"] = 0.0
                self.state[p]["was_reinitialized"] = False

    def step(self, loss = None):

        for group in self.param_groups:
            lr = group["lr"]
            beta_utility = group["beta_utility"]
            weight_decay = group["weight_decay"]
            l1_reg_factor = group["l1_reg_factor"]
            momentum = group["momentum"]
            maturity_threshold = group["maturity_threshold"]
            means = group["new_weights_mean"]
            stds = group["new_weights_std"]

            for i, p in enumerate(group["params"]):
                state = self.state[p]

                if p.grad is None:
                    continue

                dp = p.grad

                if weight_decay != 0.0:
                    dp = dp.add(p, alpha=weight_decay)
                if l1_reg_factor != 0.0:
                    dp = dp.add(p.detach().sign(), alpha=l1_reg_factor)

                if momentum != 0:
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(dp, alpha=1 - momentum)
                    dp = buf

                p.data.add_(dp, alpha=-lr)

                # prune and reinitialize weights
                if self.utility_function is not None:
                    self.prune_and_reinitialize(state, p, beta_utility, means[i], stds[i], maturity_threshold)

    def prune_and_reinitialize(self, state: dict, parameter: torch.Tensor, beta_utility: float, mean: float, std: float,
                               maturity_threshold: int):

        state["age"] += 1
        mature_weights_indices = torch.where(state["age"] > maturity_threshold)[0]
        if mature_weights_indices.size()[0] == 0: return  # weights are not mature enough

        state["current_num_reinit"] += state["num_reinit_per_step"]
        if state["current_num_reinit"] < 1.0: return            # nothing to reinitialize yet

        # compute utility
        utility = self.utility_function(parameter)
        if beta_utility != 0.0:                                 # if using moving average
            state["utility_trace"] *= beta_utility
            state["utility_trace"] += (1 - beta_utility) * utility
            utility = state["utility_trace"]
        mature_weights_utility = utility.view(-1)[state["age"] > maturity_threshold]

        # prune weights
        remainder, num_to_prune = modf(state["current_num_reinit"])
        pruned_indices = prune_number_of_weights(parameter, int(num_to_prune), mature_weights_utility, mature_weights_indices)

        # reinitialize weights
        reinitialize_weights_with_noise(parameter, pruned_indices, mean, std)

        # update buffers
        if beta_utility != 0.0:
            state["utility_trace"][pruned_indices] = 0.0
        state["age"][pruned_indices] = 0
        state["current_num_reinit"] = remainder
        state["was_reinitialized"] = True

    def reset_reinitialized_indicator(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["was_reinitialized"] = False
