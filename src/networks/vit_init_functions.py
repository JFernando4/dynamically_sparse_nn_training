import torch
import math
from .torchvision_modified_vit import CustomMLPBlock, VisionTransformer, EncoderBlock


def initialize_vit(network: VisionTransformer):
    """
    Initializes a visual transformer
    :param network: an instance of torchvision VisionTransformer
    :return: None, but initializes the weights of the transformer model
    """

    torch.nn.init.zeros_(network.class_token)
    torch.nn.init.normal_(network.encoder.pos_embedding, std=0.02)
    network.apply(xavier_vit_initialization)
    initialize_vit_heads(network.heads)


def xavier_vit_initialization(m: torch.nn.Module):
    """
    Initializes the layers of a visual transformer except for the last layer
    """
    if isinstance(m, torch.nn.Conv2d):
        fan_in = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
        torch.nn.init.trunc_normal_(m.weight, std=torch.math.sqrt(1 / fan_in))
        torch.nn.init.zeros_(m.bias)
    elif isinstance(m, torch.nn.LayerNorm):
        initialize_layer_norm_module(m)
    elif isinstance(m, EncoderBlock):
        initialize_multihead_self_attention_module(m.self_attention)
        initialize_mlp_block(m.mlp)
    else:
        return


def initialize_layer_norm_module(m: torch.nn.Module):
    """
    Initializes the weights of a layer norm module to one and the bias to zero
    """
    if not isinstance(m, torch.nn.LayerNorm): return
    if not m.elementwise_affine: return
    torch.nn.init.ones_(m.weight)
    torch.nn.init.zeros_(m.bias)


def initialize_multihead_self_attention_module(m: torch.nn.Module):
    """
    Initializes a multihead attention module using xavier normal initialization
    """
    if not isinstance(m, torch.nn.MultiheadAttention): return

    if m._qkv_same_embed_dim:
        torch.nn.init.xavier_uniform_(m.in_proj_weight)
    else:
        torch.nn.init.xavier_uniform_(m.q_proj_weight)
        torch.nn.init.xavier_uniform_(m.k_proj_weight)
        torch.nn.init.xavier_uniform_(m.v_proj_weight)

    # Torchvision encoder block doesn't directly initialize the out_proj.weight, but uses the default initialization
    # of a linear layer. Strangely, torchvision does initialize the out_proj.bias
    torch.nn.init.kaiming_uniform_(m.out_proj.weight, a=math.sqrt(5))

    if m.in_proj_bias is not None:
        torch.nn.init.zeros_(m.in_proj_bias)
        torch.nn.init.zeros_(m.out_proj.bias)
    if m.bias_k is not None:
        torch.nn.init.xavier_normal_(m.bias_k)
    if m.bias_v is not None:
        torch.nn.init.xavier_normal_(m.bias_v)


def initialize_mlp_block(m: torch.nn.Module):
    """
    Initializes a visual transformer encoder block's mlp block
    """

    if not isinstance(m, CustomMLPBlock): return

    # This is what I ended up doing by accident, which surprisingly results in 58.9% accuracy
    # for sub_m in m.modules():
    #     if isinstance(sub_m, torch.nn.Linear):
    #         torch.nn.init.kaiming_uniform_(sub_m.weight, a=math.sqrt(5))
    #         if sub_m.bias is not None:
    #             fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(sub_m.weight)
    #             bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    #             torch.nn.init.uniform_(sub_m.bias, -bound, bound)

    # This is how torchvision does it, which give about 57.4% accuracy in CIFAR-100
    for sub_m in m.modules():
        if isinstance(sub_m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(sub_m.weight)
            if sub_m.bias is not None:
                torch.nn.init.normal_(sub_m.bias, std=1e-6)


def initialize_vit_heads(m: torch.nn.Sequential):
    """
    Initializes the classification heads of a visual transformer
    """

    torch.nn.init.zeros_(m[0].weight)
    torch.nn.init.zeros_(m[0].bias)

    if len(m) > 1:
        raise ValueError("Don't know how to handle heads with a representation layer.")

