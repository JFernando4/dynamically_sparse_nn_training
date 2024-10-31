from typing import Union
import torch

from .weight_matrix_updates import setup_cbpw_weight_update_function
from src.networks.torchvision_modified_vit import VisionTransformer, EncoderBlock
from src.networks.torchvision_modified_resnet import ResNet, BasicBlock
from src.networks.permuted_mnist_network import ThreeHiddenLayerNetwork


def initialize_weight_dict(net: torch.nn.Module,
                           architecture_type: str,
                           prune_method: str,
                           grow_method: str,
                           drop_factor: float,
                           **kwargs) -> dict[str, tuple]:
    """ Initializes the weight dictionaries used in CBPw """

    if architecture_type == "vit":
        assert isinstance(net, VisionTransformer)
        ln_drop_factor = drop_factor if "ln_drop_factor" not in kwargs.keys() else kwargs["ln_drop_factor"]
        return initialize_weights_dict_df_as_rate(net, prune_method, grow_method, drop_factor, ln_drop_factor=ln_drop_factor)

    elif architecture_type == "resnet":
        assert isinstance(net, ResNet)
        noise_std = None if "noise_std" not in kwargs.keys() else kwargs["noise_std"]
        ln_drop_factor = drop_factor if "ln_drop_factor" not in kwargs.keys() else kwargs["ln_drop_factor"]
        return initialize_weights_dict_df_as_rate(net, prune_method, grow_method, drop_factor,
                                                  ln_drop_factor=ln_drop_factor, noise_std=noise_std)

    elif architecture_type == "sequential":
        assert isinstance(net, ThreeHiddenLayerNetwork)
        noise_std = None if "noise_std" not in kwargs.keys() else kwargs["noise_std"]
        return initialize_weights_dict_sequential(net, prune_method=prune_method, grow_method=grow_method,
                                                  drop_factor=drop_factor, noise_std=noise_std)
    elif architecture_type == "bert":
        return initialize_weights_dict_bert_all(net, prune_method=prune_method, grow_method=grow_method, drop_factor=drop_factor)
    else:
        raise ValueError(f"{architecture_type} is not a valid architecture type.")


def initialize_bn_list_resnet(net: ResNet, exclude_downsample: bool = False):
    """
    Returns a list with all the BatchNormalization layers in a ResNet model
    """
    list_of_batch_norm_layers = [net.bn1]

    for residual_stack in (net.layer1, net.layer2, net.layer3, net.layer4):
        for residual_block in residual_stack:
            assert isinstance(residual_block, BasicBlock)
            list_of_batch_norm_layers.append(residual_block.bn1)
            list_of_batch_norm_layers.append(residual_block.bn2)
            if (residual_block.downsample is not None) and (not exclude_downsample):
                list_of_batch_norm_layers.append(residual_block.downsample[1])

    return list_of_batch_norm_layers


def initialize_ln_list_bert(net):
    """
    Returns a list with all the LayerNormalization layers in a pretrained BERT model
    """
    list_of_layer_norm_layers = [net.bert.embeddings.LayerNorm]

    for i in range(4):
        list_of_layer_norm_layers.append(net.bert.encoder.layer[i].output.LayerNorm)

    return list_of_layer_norm_layers


def initialize_weights_dict_df_as_rate(net: Union[VisionTransformer, ResNet],
                                       prune_method: str,
                                       grow_method: str,
                                       drop_factor: float,
                                       ln_drop_factor: float,
                                       noise_std: float = None) -> dict[str, tuple]:
    """
    Initializes the weight dictionaries used in CBPw for a network. The drop_factor is used as a rate, which
    is relevant if drop_factor * p.numel() is less than 1.
    """
    bias_grow_name = "zero" if grow_method != "fixed_with_noise" else grow_method
    ln_weight_grow_name = "fixed" if grow_method != "fixed_with_noise" else grow_method
    weight_update_func = setup_cbpw_weight_update_function(prune_method, grow_method, drop_factor=drop_factor,
                                                            as_rate=True, reinit_val=0.0, noise_std=noise_std)
    bias_update_func = setup_cbpw_weight_update_function(prune_method, grow_name=bias_grow_name, drop_factor=drop_factor,
                                                         as_rate=True, reinit_val=0.0, noise_std=noise_std)
    ln_weight_update_func = setup_cbpw_weight_update_function(prune_method, grow_name=ln_weight_grow_name,
                                                              drop_factor=ln_drop_factor, as_rate=True, reinit_val=1.0,
                                                              noise_std=noise_std)

    weight_dict = {}
    for n, p in net.named_parameters():
        is_weight = "weight" in n
        is_bias = "bias" in n
        is_layer_or_batch_norm = (".ln_1." in n) or (".ln_2." in n) or (".ln." in n) or ("bn1." in n) or ("bn2." in n) or ("downsample.1." in n)

        if is_weight and is_layer_or_batch_norm:
            weight_dict[n] = (p, ln_weight_update_func)
        elif is_bias:
            weight_dict[n] = (p, bias_update_func)
        else:
            weight_dict[n] = (p, weight_update_func)

    return weight_dict


def initialize_weights_dict_sequential(net: ThreeHiddenLayerNetwork,
                                       prune_method: str,
                                       grow_method: str,
                                       drop_factor: float,
                                       noise_std: float = None) -> dict[str, tuple]:
    """ Initializes the weight dictionaries used in CBPw for a Sequential Network """
    bias_grow_name = "zero" if grow_method != "fixed_with_noise" else grow_method
    ln_weight_grow_name = "fixed" if grow_method != "fixed_with_noise" else grow_method
    weights_update_func = setup_cbpw_weight_update_function(prune_method, grow_method, drop_factor=drop_factor,
                                                            as_rate=True, reinit_val=0.0, noise_std=noise_std, activation="relu")
    bias_update_func = setup_cbpw_weight_update_function(prune_method, grow_name=bias_grow_name, drop_factor=drop_factor,
                                                         as_rate=True, reinit_val=0.0, noise_std=noise_std)
    ln_weight_update_func = setup_cbpw_weight_update_function(prune_method, grow_name=ln_weight_grow_name,
                                                              drop_factor=drop_factor, as_rate=True, reinit_val=1.0,
                                                              noise_std=noise_std)

    weight_dict = {}

    for n, p in net.named_parameters():
        is_weight = "weight" in n
        is_bias = "bias" in n
        is_layer_norm = "ln_" in n

        if is_weight and is_layer_norm:
            weight_dict[n] = (p, ln_weight_update_func)
        elif is_bias:
            weight_dict[n] = (p, bias_update_func)
        else:
            weight_dict[n] = (p, weights_update_func)

    return weight_dict


def initialize_weights_dict_bert_all(net, prune_method: str, grow_method: str, drop_factor: float):
    """
    Initializes the weight dictionary required for CBPw for a Bert model
    """
    update_func = setup_cbpw_weight_update_function(prune_method, grow_method, drop_factor=drop_factor, as_rate=True)
    bias_func = setup_cbpw_weight_update_function(prune_method, grow_name="zero", drop_factor=drop_factor, as_rate=True)
    layer_norm_weight_func = setup_cbpw_weight_update_function(prune_method, grow_name="fixed", drop_factor=drop_factor, reinit_val=1.0, as_rate=True)
    weight_dict = {}

    for n, p in net.named_parameters():
        is_layer_norm = "LayerNorm" in n
        is_bias = "bias" in n
        is_weight = "weight" in n

        if is_layer_norm and is_weight:
            temp_update_func = layer_norm_weight_func
        elif is_bias:
            temp_update_func = bias_func
        else:
            temp_update_func = update_func

        weight_dict[n] = (p, temp_update_func)

    return weight_dict
