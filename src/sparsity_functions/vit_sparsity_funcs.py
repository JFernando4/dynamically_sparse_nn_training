# third party libraries
from torchvision.models.vision_transformer import VisionTransformer, EncoderBlock, MLPBlock
# src
from sparsity_funcs import init_weight_mask_from_tensor


def init_vit_weight_masks(net: VisionTransformer, sparsity_level: float):
    """
    Initializes the weight masks for vision transformers not including layer norm modules

    Args:
        net: VisionTransformer class instance
        sparsity_level: float between (0,1) indicating the sparisity level

    Returns
        list of dictionaries for each weight matrix in the vision transformer
    """

    masks = []
    conv_proj_mask = init_weight_mask_from_tensor(net.conv_proj.weight, sparsity_level)
    masks.append(conv_proj_mask)

    pos_embedding_mask = init_weight_mask_from_tensor(net.encoder.pos_embedding, sparsity_level)
    masks.append(pos_embedding_mask)

    # encoder_masks = get_vit_encoder_masks()


