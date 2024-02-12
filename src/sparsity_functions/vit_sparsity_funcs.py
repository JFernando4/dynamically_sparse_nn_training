# third party libraries
import torch.nn
from torch.nn.init import _calculate_fan_in_and_fan_out
from torchvision.models.vision_transformer import VisionTransformer, EncoderBlock, MLPBlock, Encoder
# src
from src.sparsity_functions.sparsity_funcs import init_weight_mask_from_tensor


def get_xavier_uniform_init_std(tensor: torch.Tensor()):
    """ Returns the standard deviation used by xavier initialization on a given tensor """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return torch.math.sqrt(2.0 / (fan_in + fan_out))


def init_vit_weight_masks(net: VisionTransformer, sparsity_level: float, include_head: bool = False,
                          include_class_token: bool = False, include_pos_embedding: bool = False):
    """
    Initializes the weight masks for vision transformers not including layer norm modules

    Args:
        net: VisionTransformer class instance
        sparsity_level: float between [0,1) indicating the sparsity level
        include_head: bool indicating whether to also generate a mask for the head of the network
        include_class_token: bool indicating whether to also generate a mask for the class token parameter
        include_pos_embedding: bool indicating whether to also generate a mask for the pos_embedding parameter

    Returns
        list of dictionaries for each weight matrix in the vision transformer
    """

    masks = []
    # generate mask for convolutional projection
    conv_proj_mask = init_weight_mask_from_tensor(net.conv_proj.weight, sparsity_level)
    fan_in = net.conv_proj.in_channels * net.conv_proj.kernel_size[0] * net.conv_proj.kernel_size[1]
    conv_proj_mask["init_func"] = lambda z: torch.nn.init.trunc_normal_(z, torch.math.sqrt(1 / fan_in))
    conv_proj_mask["init_std"] = torch.math.sqrt(1 / fan_in)
    masks.append(conv_proj_mask)
    # generate mask for class_token parameters
    if include_class_token:
        class_token_mask = init_weight_mask_from_tensor(net.class_token, sparsity_level)
        class_token_mask["init_func"] = torch.nn.init.xavier_uniform_
        class_token_mask["init_std"] = get_xavier_uniform_init_std(net.class_token)
        masks.append(class_token_mask)
    # generate mask for pos_embedding parameters
    if include_pos_embedding:
        pos_embedding_mask = init_weight_mask_from_tensor(net.encoder.pos_embedding, sparsity_level)
        pos_embedding_mask["init_func"] = lambda z: torch.nn.init.normal_(z, std=0.02)
        pos_embedding_mask["init_std"] = 0.02
        masks.append(pos_embedding_mask)
    # generate masks for encoder
    masks.extend(init_vit_encoder_masks(net.encoder, sparsity_level))
    # generate masks for head of the network
    if include_head:
        head_mask = init_weight_mask_from_tensor(net.heads[0].weight, sparsity_level)
        head_mask["init_func"] = torch.nn.init.zeros_
        masks.append(head_mask)

    return masks


def init_vit_encoder_masks(mod: Encoder, sparsity_level: float):
    """
    Initializes the weights masks for the encoder of the vision transformer

     Args:
         mod: instance of torchvision's Encoder
         sparsity_level: float in [0,1) indicating the sparsity level

    Returns:
        list of dictionaries for each weight matrix of each layer in the encoder
    """

    masks = []

    for e_block in mod.layers:
        assert isinstance(e_block, EncoderBlock)
        masks.extend(init_encoder_block_masks(e_block, sparsity_level))

    return masks


def init_encoder_block_masks(mod: EncoderBlock, sparsity_level: float):
    """
    Initializes the weights masks for the encoder blocks

     Args:
         mod: instances of torchvision's EncoderBlock
         sparsity_level: float in [0,1) indicating the sparsity level

    Returns:
        list of dictionaries for each weight matrix of each layer in the encoder
    """
    masks = []

    # multi-head attention masks
    in_proj_mask = init_weight_mask_from_tensor(mod.self_attention.in_proj_weight, sparsity_level)
    in_proj_mask["init_func"] = torch.nn.init.xavier_uniform_
    in_proj_mask["init_std"] = get_xavier_uniform_init_std(mod.self_attention.in_proj_weight)
    masks.append(in_proj_mask)

    out_proj_mask = init_weight_mask_from_tensor(mod.self_attention.in_proj_weight, sparsity_level)
    out_proj_mask["init_func"] = torch.nn.init.xavier_uniform_
    out_proj_mask["init_std"] = get_xavier_uniform_init_std(mod.self_attention.out_proj.weight)
    masks.append(out_proj_mask)

    # mlp block masks
    for sub_m in mod.mlp.modules():
        if isinstance(sub_m, torch.nn.Linear):
            temp_mask = init_weight_mask_from_tensor(sub_m.weight, sparsity_level)
            temp_mask["init_func"] = torch.nn.init.xavier_uniform_
            temp_mask["init_std"] = get_xavier_uniform_init_std(sub_m.weight)
            masks.append(temp_mask)

    return masks

