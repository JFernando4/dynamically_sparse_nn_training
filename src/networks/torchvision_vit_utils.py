import torch
from torchvision.models.vision_transformer import VisionTransformer, EncoderBlock, MLPBlock


def initialize_vit(network: VisionTransformer):
    """
    Initializes a visual transformer
    :param network: an instance of torchvision VisionTransformer
    :return: None, but initializes the weights of the transformer model
    """

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
    elif isinstance(m, EncoderBlock):
        initialize_layer_norm_module(m.ln_1)
        initialize_self_multihead_attention_module(m.self_attention)
        initialize_layer_norm_module(m.ln_2)
        initialize_mlp_block(m.mlp)
    else:
        return


def initialize_layer_norm_module(m: torch.nn.LayerNorm):
    """
    Initializes the weights of a layer norm module to one and the bias to zero
    """
    torch.nn.init.ones_(m.weight)
    torch.nn.init.zeros_(m.bias)


def initialize_self_multihead_attention_module(m: torch.nn.MultiheadAttention):
    """
    Initializes a multihead attention module using xavier normal initialization
    """
    if m._qkv_same_embed_dim:
        torch.nn.init.xavier_uniform_(m.in_proj_weight)
    else:
        torch.nn.init.xavier_uniform_(m.q_proj_weight)
        torch.nn.init.xavier_uniform_(m.k_proj_weight)
        torch.nn.init.xavier_uniform_(m.v_proj_weight)

    if m.in_proj_bias is not None:
        torch.nn.init.zeros_(m.in_proj_bias)
        torch.nn.init.zeros_(m.out_proj.bias)
    if m.bias_k is not None:
        torch.nn.init.xavier_normal_(m.bias_k)
    if m.bias_v is not None:
        torch.nn.init.xavier_normal_(m.bias_v)


def initialize_mlp_block(m: MLPBlock):
    """
    Initializes a visual transformer encoder block's mlp block
    """
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
