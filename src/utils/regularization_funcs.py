# third party libraries
import torch

# mlproj_manager improts
from mlproj_manager.util import apply_regularization_to_tensor


@torch.no_grad()
def apply_regularization_to_sequential_net(net: torch.nn.Sequential, l2_factor: float = 0.0, l1_factor: float = 0.0,
                                           regularized_bias: bool = False):
    """
    Applies regularization to each module in a Sequential module
    :param net: an instance of torch.nn.Sequential
    :param l2_factor: float corresponding to the l2 penalty
    :param l1_factor: float corresponding to the l1 penalty
    :param regularized_bias: bool indicating whether to regularized bias
    :return None, but changes the values of the weights and bias in each module of the network
    """

    for mod in list(net):
        if hasattr(mod, "weight"):
            apply_regularization_to_tensor(mod.weight, l1_factor=l1_factor, l2_factor=l2_factor)
        if hasattr(mod, "bias") and regularized_bias:
            apply_regularization_to_tensor(mod.bias, l1_factor=l1_factor, l2_factor=l2_factor)


@torch.no_grad()
def apply_regularization_to_net(net: torch.nn.Module, l2_factor: float = 0.0, l1_factor: float = 0.0):
    """
    Applies regularization to all the parameters in a network
    :param net: instance of torch.nn.Module
    :param l2_factor: float corresponding to the l2 penalty
    :param l1_factor: float corresponding to the l1 penalty
    :return: None, but changes the values of all the parameters in the network
    """

    for param in net.parameters():
        apply_regularization_to_tensor(param.data, l1_factor=l1_factor, l2_factor=l2_factor)


def apply_regularization_to_parameter_list(parameter_list, scale_factor: float = 1.0):
    """
    Regularizes all the parameters in a given list. Each entry in the list is a dictionary with three key-value pairs:
        "parameter" - parameter to be regularized
        "l1" - l1-regularization factor
        "l2" - l2-regularization factor

    :param parameter_list: list of dictionaries
    :param scale_factor: positive float to scale the l1 and l2 factor by
    """

    for parameter_dict in parameter_list:
        if parameter_dict["l1"] > 0.0 or parameter_dict["l2"] > 0.0:
            temp_l1 = parameter_dict["l1"] * scale_factor
            temp_l2 = parameter_dict["l2"] * scale_factor
            apply_regularization_to_tensor(parameter_dict["parameter"], temp_l1, temp_l2)
