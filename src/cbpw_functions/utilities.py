
import torch

from .weight_matrix_updates import setup_cbpw_weight_update_function
from src.networks.torchvision_modified_vit import VisionTransformer, EncoderBlock
from src.networks.torchvision_modified_resnet import ResNet, BasicBlock


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
        exclude_downsample = False if "exclude_downsample" not in kwargs.keys() else kwargs["exclude_downsample"]
        include_output_layer = False if "include_output_layer" not in kwargs.keys() else kwargs["include_output_layer"]
        include_all = False if "include_all" not in kwargs.keys() else kwargs["include_all"]
        return initializes_weights_dict_resnet(net, prune_method=prune_method, grow_method=grow_method,
                                               drop_factor=drop_factor, exclude_downsample=exclude_downsample,
                                               include_output_layer=include_output_layer, include_all=include_all)

    elif architecture_type == "sequential":
        assert isinstance(net, torch.nn.Sequential)
        return initialize_weights_dict_sequential(net, prune_method=prune_method, grow_method=grow_method,
                                                  drop_factor=drop_factor)
    elif architecture_type == "bert":
        return initialize_weights_dict_bert(net, prune_method=prune_method, grow_method=grow_method, drop_factor=drop_factor)
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


def initialize_ln_list_vit(net: VisionTransformer):
    """
    Returns a list with all the LayerNormalization layers in a VisionTransformer model
    """
    list_of_layer_norm_layers = []

    for encoder_block in list(net.encoder.layers):
        assert isinstance(encoder_block, EncoderBlock)
        list_of_layer_norm_layers.extend([encoder_block.ln_1, encoder_block.ln_2])
    list_of_layer_norm_layers.append(net.encoder.ln)

    return list_of_layer_norm_layers


def initialize_ln_list_bert(net):
    """
    Returns a list with all the LayerNormalization layers in a pretrained BERT model
    """
    list_of_layer_norm_layers = [net.bert.embeddings.LayerNorm]

    for i in range(4):
        list_of_layer_norm_layers.append(net.bert.encoder.layer[i].output.LayerNorm)

    return list_of_layer_norm_layers


def initialize_weights_dict_vit(net: VisionTransformer,
                                prune_method: str,
                                grow_method: str,
                                drop_factor: float,
                                include_class_token: bool,
                                include_conv_proj: bool,
                                include_pos_embedding: bool,
                                include_self_attention: bool,
                                include_head: bool) -> dict[str, tuple]:
    """ Initializes the weight dictionaries used in CBPw for a Vision Transformer"""

    update_func = setup_cbpw_weight_update_function(prune_method, grow_method, drop_factor=drop_factor)

    weight_dict = {}
    for n, p in net.named_parameters():
        if "class_token" in n and include_class_token:
            weight_dict[n] = (p, update_func)
        if "conv_proj.weight" in n and include_conv_proj:
            weight_dict[n] = (p, update_func)
        if "pos_embedding" in n and include_pos_embedding:
            weight_dict[n] = (p, update_func)
        if ("in_proj_weight" in n or "out_proj.weight" in n or ("mlp" in n and "weight" in n)) and include_self_attention:
            weight_dict[n] = (p, update_func)
        if ("head.weight" in n) and include_head:
            weight_dict[n] = (p, update_func)

    return weight_dict


def initializes_weights_dict_resnet(net: ResNet,
                                    prune_method: str,
                                    grow_method: str,
                                    drop_factor: float,
                                    exclude_downsample: bool,
                                    include_output_layer: bool,
                                    include_all: bool = False) -> dict[str, tuple]:
    """ Initializes the weight dictionaries used in CBPw for a Residual Network"""
    update_func = setup_cbpw_weight_update_function(prune_method, grow_method, drop_factor=drop_factor)

    weight_dict = {}
    if include_all:
        for n, p in net.named_parameters():
            is_bn_layer = ("bn" in n) or ("downsample.1" in n)
            if p.requires_grad and not is_bn_layer:
                weight_dict[n] = (p, update_func)
        return weight_dict

    for n, p in net.named_parameters():
        if "conv" in n and "weight" in n:
            weight_dict[n] = (p, update_func)
        if ("downsample.0" in n and "weight" in n) and not exclude_downsample:
            weight_dict[n] = (p, update_func)
        if ("fc.weight" in n) and include_output_layer:
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


def initialize_weights_dict_bert(net, prune_method: str, grow_method: str, drop_factor: float):
    """
    Initializes the weight dictionary required for CBPw for a Bert model
    """
    update_func = setup_cbpw_weight_update_function(prune_method, grow_method, drop_factor=drop_factor)
    weight_dict = {}

    for n, p in net.named_parameters():
        is_weight_matrix = ".weight" in n
        is_not_layer_norm = "LayerNorm" not in n
        if is_weight_matrix and is_not_layer_norm:
            weight_dict[n] = (p, update_func)

    return weight_dict
