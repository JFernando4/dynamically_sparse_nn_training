
import torch

from .weight_matrix_updates import setup_cbpw_weight_update_function
from src.networks.torchvision_modified_vit import VisionTransformer
from src.networks.torchvision_modified_resnet import ResNet


def initialize_weight_dict(net: torch.nn.Module,
                           architecture_type: str,
                           prune_method: str,
                           grow_method: str,
                           drop_factor: float,
                           **kwargs) -> dict[str, tuple]:
    """ Initializes the weight dictionaries used in CBPw """

    if architecture_type == "vit":
        assert isinstance(net, VisionTransformer)
        return initialize_weights_dict_vit(net, prune_method=prune_method, grow_method=grow_method,
                                           drop_factor=drop_factor, **kwargs)

    elif architecture_type == "resnet":
        assert isinstance(net, ResNet)
        return initializes_weights_dict_resnet(net, prune_method=prune_method, grow_method=grow_method,
                                               drop_factor=drop_factor, **kwargs)

    elif architecture_type == "sequential":
        assert isinstance(net, torch.nn.Sequential)
        return initialize_weights_dict_sequential(net, prune_method=prune_method, grow_method=grow_method,
                                                  drop_factor=drop_factor)

    else:
        raise ValueError(f"{architecture_type} is not a valid architecture type.")


def initialize_weights_dict_vit(net: VisionTransformer,
                                prune_method: str,
                                grow_method: str,
                                drop_factor: float,
                                include_class_token: bool,
                                include_conv_proj: bool,
                                include_pos_embedding: bool,
                                include_layer_norm: bool,
                                include_self_attention: bool) -> dict[str, tuple]:
    """ Initializes the weight dictionaries used in CBPw for a Vision Transformer"""

    update_func = setup_cbpw_weight_update_function(prune_method, grow_method, drop_factor=drop_factor)
    ln_update_func = setup_cbpw_weight_update_function(prune_method, "fixed", reinit_val=1.0, drop_factor=drop_factor)

    weight_dict = {}
    for n, p in net.named_parameters():
        if "class_token" in n and include_class_token:
            weight_dict[n] = (p, update_func)
        if "conv_proj.weight" in n and include_conv_proj:
            weight_dict[n] = (p, update_func)
        if "pos_embedding" in n and include_pos_embedding:
            weight_dict[n] = (p, update_func)
        if ("ln" in n and "weight" in n) and include_layer_norm:
            weight_dict[n] = (p, ln_update_func)
        if ("in_proj_weight" in n or "out_proj.weight" in n or ("mlp" in n and "weight" in n)) and include_self_attention:
            weight_dict[n] = (p, update_func)

    return weight_dict


def initializes_weights_dict_resnet(net: ResNet,
                                    prune_method: str,
                                    grow_method: str,
                                    drop_factor: float,
                                    include_bn: bool) -> dict[str, tuple]:
    """ Initializes the weight dictionaries used in CBPw for a Residual Network"""
    update_func = setup_cbpw_weight_update_function(prune_method, grow_method, drop_factor=drop_factor)

    weight_dict = {}

    for n, p in net.named_parameters():
        if ("conv" in n and "weight" in n) or ("downsample.0" in n and "weight" in n):
            weight_dict[n] = (p, update_func)
        if (("bn" in n and "weight" in n) or ("downsample.1" in n and "weight" in n)) and include_bn:
            weight_dict[n] = (p, update_func)
    return weight_dict


def initialize_weights_dict_sequential(net: torch.nn.Sequential,
                                       prune_method: str,
                                       grow_method: str,
                                       drop_factor: float) -> dict[str, tuple]:
    """ Initializes the weight dictionaries used in CBPw for a Sequential Network """
    update_func = setup_cbpw_weight_update_function(prune_method, grow_method, drop_factor=drop_factor)

    weight_dict = {}

    layer_index = 0
    for i in range(len(net) - 1):
        if isinstance(net[i], torch.nn.Linear):
            weight_dict[f"linear_{layer_index}"] = (net[i].weight, update_func)
            layer_index += 1

    return weight_dict
