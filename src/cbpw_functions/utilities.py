
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
        df_as_rate = False if "df_as_rate" not in kwargs else kwargs["df_as_rate"]
        if df_as_rate:
            return initialize_weights_dict_vit_df_as_rate(net, prune_method, grow_method, drop_factor)
        else:
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


def initialize_weights_dict_vit_df_as_rate(net: VisionTransformer,
                                           prune_method: str,
                                           grow_method: str,
                                           drop_factor: float) -> dict[str, tuple]:
    """
    Initializes the weight dictionaries used in CBPw for a Vision Transformer. The drop_factor is used as a rate, which
    is relevant if drop_factor * p.numel() is less than 1.
    """
    weight_update_func = setup_cbpw_weight_update_function(prune_method, grow_method, drop_factor=drop_factor, as_rate=True)
    bias_update_func = setup_cbpw_weight_update_function(prune_method, "zero", drop_factor=drop_factor, as_rate=True)
    ln_weight_update_func = setup_cbpw_weight_update_function(prune_method, grow_name="fixed", drop_factor=drop_factor,
                                                              as_rate=True, reinit_val=1.0)

    weight_dict = {}
    for n, p in net.named_parameters():
        is_weight = "weight" in n
        is_bias = "bias" in n
        is_layer_norm = (".ln_1." in n) or (".ln_2." in n) or (".ln." in n)

        if is_weight and is_layer_norm:
            weight_dict[n] = (p, ln_weight_update_func)
        elif is_bias:
            weight_dict[n] = (p, bias_update_func)
        else:
            weight_dict[n] = (p, weight_update_func)

    return weight_dict


def initialize_weights_dict_vit(net: VisionTransformer,
                                prune_method: str,
                                grow_method: str,
                                drop_factor: float,
                                include_class_token: bool,
                                include_conv_proj: bool,
                                include_pos_embedding: bool,
                                include_self_attention: bool,
                                include_head: bool) -> dict[str, tuple]:
    """ Initializes the weight dictionaries used in CBPw for a Vision Transformer """

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
    weights_update_func = setup_cbpw_weight_update_function(prune_method, grow_method, drop_factor=drop_factor, as_rate=True)
    bias_update_func = setup_cbpw_weight_update_function(prune_method, grow_name="zero", drop_factor=drop_factor, as_rate=True)

    weight_dict = {}

    for n, p in net.named_parameters():
        if "bias" in n:
            temp_update_func = bias_update_func
        else:
            temp_update_func = weights_update_func
        weight_dict[n] = (p, temp_update_func)

    return weight_dict


def initialize_weights_dict_bert(net, prune_method: str, grow_method: str, drop_factor: float,  exclude_embeddings:bool):
    """
    Initializes the weight dictionary required for CBPw for a Bert model

    params:
        exclude_embeddings: bool indicating whether to omit the word, position, and token_type embeddings
    """
    update_func = setup_cbpw_weight_update_function(prune_method, grow_method, drop_factor=drop_factor)
    weight_dict = {}

    for n, p in net.named_parameters():
        if exclude_embeddings:
            is_word_embedding = "word_embeddings" in n
            is_position_embedding = "position_embeddings" in n
            is_token_type_embedding = "token_type_embeddings" in n
            if is_word_embedding or is_position_embedding or is_token_type_embedding: continue

        is_weight_matrix = ".weight" in n
        is_not_layer_norm = "LayerNorm" not in n
        if is_weight_matrix and is_not_layer_norm:
            weight_dict[n] = (p, update_func)

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
