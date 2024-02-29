import torch
from torchvision.models.vision_transformer import VisionTransformer, EncoderBlock, MLPBlock


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
        initialize_self_multihead_attention_module(m.self_attention)
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

    torch.nn.init.xavier_uniform_(m.out_proj.weight)

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


def init_weight_regularization_list(net: VisionTransformer, l1_factor: float = 0.0,
                                    apply_l1_reg_ct: bool = False, apply_l1_reg_pe: bool = False,
                                    apply_l1_reg_cp: bool = False, apply_l1_reg_msa: bool = False):
    """
    Initializes a dictionary with the weight matrices that need to be regularized and the corresponding l1 and l2
    regularization factors. L2-regularization is applied to every parameter in the network, whereas L1-regularization
    is only applied to the class token, positional embedding, convolutional projection, and multi-head self-attention
    if their corresponding flags are set to True. L1-regularization is always applied to the MLP blocks as long as
    l1_factor is greater than zero. L1-regularization is only applied to weight matrices.

    :param net: instance of torchvision VisionTransformer class
    :param l1_factor: l1-regularization penalty
    :param apply_l1_reg_ct: bool indicating whether to apply l1-regularization to the class token
    :param apply_l1_reg_pe: bool indicating whether to apply l1-regularization to the positional embedding
    :param apply_l1_reg_cp: bool indicating whether to apply l1-regularization to the convolutional projection
    :param apply_l1_reg_msa: bool indicating whether to apply l1-regularization to attention layers
    :return: list with an entry for each layer
             each entry in the list is a dictionary with keys: ["parameter", "l1"]
    """
    assert 0.0 <= l1_factor

    regularization_list = []

    # class token parameters
    ct_l1_penalty = l1_factor if apply_l1_reg_ct else 0.0
    class_token_dict = {"parameter": net.class_token, "l1": ct_l1_penalty}
    regularization_list.append(class_token_dict)

    # convolutional projection parameters
    cp_l1_penalty = l1_factor if apply_l1_reg_cp else 0.0
    conv_proj_weight_dict = {"parameter": net.conv_proj.weight, "l1": cp_l1_penalty}
    regularization_list.extend([conv_proj_weight_dict])

    # positional embedding parameters
    pe_l1_penalty = l1_factor if apply_l1_reg_pe else 0.0
    pos_embedding_dict = {"parameter": net.encoder.pos_embedding, "l1": pe_l1_penalty}
    regularization_list.append(pos_embedding_dict)

    # encoder parameters
    for encoder_block in net.encoder.layers:
        temp_list = get_encoder_block_reg_dictionaries(encoder_block, l1_factor, apply_l1_reg_msa)
        regularization_list.extend(temp_list)

    return regularization_list


def get_encoder_block_reg_dictionaries(mod: EncoderBlock, l1_factor: float = 0.0, apply_l1_reg_msa: bool = False):
    """
    Gets the parameters that need to be regularized in the EncoderBlock.
    See function above for the description of the arguments.
    """

    regularization_list = []

    msa_l1_penalty = l1_factor if apply_l1_reg_msa else 0.0
    in_proj_weight_dict = {"parameter": mod.self_attention.in_proj_weight, "l1": msa_l1_penalty}
    out_proj_weight_dict = {"parameter": mod.self_attention.out_proj.weight, "l1": msa_l1_penalty}
    regularization_list.extend([in_proj_weight_dict, out_proj_weight_dict])

    for sub_m in mod.mlp.modules():
        if isinstance(sub_m, torch.nn.Linear):
            temp_weight_dict = {"parameter": sub_m.weight, "l1": l1_factor}
            regularization_list.extend([temp_weight_dict])

    return regularization_list

