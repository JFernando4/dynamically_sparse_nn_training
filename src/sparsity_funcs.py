import torch
import wandb


def update_one_weight_mask_set(mask, weight, refresh_num, reinit='zero'):
    """Updates the weight mask of one layer.
    SET algorithm: https://arxiv.org/abs/1707.04780
    (but with standard magnitude pruning)

    Args:
        mask: The weight mask.
        weight: The weights of one layer, corresponding to the mask.
        refresh_num: The number of weights to drop and grow.
        reinit: How to reinitialize the weights that are regrown. Options: 'zero', 'kaiming_normal'
        """
    mask = prune_magnitude(mask, weight, refresh_num)
    mask = grow_random(mask, weight, refresh_num, reinit)
    return mask


def prune_magnitude(mask, weight, drop_num):
    """Prunes the weight mask by dropping the smallest magnitude weights."""
    weight = torch.abs(weight) + 1e-3  # if active weights happen to be 0, they will be dropped by this 1e-3 constant
    weight = weight * mask  # only consider active weights
    sorted_weights, indices = torch.sort(weight.flatten())
    # find first non-zero weight
    for i, w in enumerate(sorted_weights):
        if w > 0:
            break
    set_zero = min(i + drop_num, len(indices))
    indices = indices[:set_zero]
    mask.view(-1)[indices] = 0.
    return mask


def grow_random(mask, weight, grow_num, reinit):
    """Grows connections in the weight mask by randomly selecting inactive weights."""
    indices = torch.where(mask.flatten() == 0)[0]
    non_active = len(indices)
    if non_active == 0:
        return mask
    shuffle = torch.randperm(non_active)
    indices = indices[shuffle]
    to_grow = min(grow_num, non_active)
    indices = indices[:to_grow]
    mask.view(-1)[indices] = 1.
    if reinit != 'zero':
        reinit_grown_weights(weight, indices, reinit)
    return mask


def reinit_grown_weights(weight, indices, reinit):
    if reinit == 'kaiming_normal':
        temp_weight = torch.empty_like(weight)
        torch.nn.init.kaiming_normal_(temp_weight)
        weight.view(-1)[indices] = temp_weight.view(-1)[indices]
    else:
        raise ValueError(f'Unknown reinit option {reinit}')


def init_weight_mask(layer, sparsity):
    """Initializes a weight mask for a layer.

    Args:
        layer: The layer to initialize the weight mask for.
        sparsity: The sparsity of the weight mask.

    Returns:
        A dict containing the mask and the weights of the layer.
    """
    num_pruned = int(layer.weight.numel() * sparsity)
    mask = torch.ones_like(layer.weight, dtype=torch.float32, requires_grad=False).to(layer.weight.device)
    mask.view(-1)[torch.randperm(layer.weight.numel())[:num_pruned]] = 0.
    return {'weight': layer.weight, 'mask': mask}


def apply_weight_masks(masks):
    """Applies the weight masks to the weights.

    Args:
        masks: list of dicts, each dict contains 'weight' and 'mask' keys
    """
    for mask_dict in masks:
        mask_dict['weight'].data *= mask_dict['mask']


def update_weight_masks(masks, drop_fraction):
    """Updates all the weight masks.

    Args:
        masks: list of dicts, each dict contains 'weight' and 'mask' keys
        drop_fraction: The fraction of weights to drop and grow.
    """
    for mask_dict in masks:
        num_active = mask_dict['mask'].sum().item()
        num_drop = int(num_active * drop_fraction)
        mask_dict['mask'] = update_one_weight_mask(mask_dict['mask'], mask_dict['weight'], num_drop)


def maintain_sparsity_target_layer(target_param, mask):
    """Maintains the sparsity of a target network layer.
    Sets the smallest weights to 0.

    Args:
        target_param: The target layer weights to maintain the sparsity of.
        mask: The weight mask of the corresponding layer in the online network.
    """
    active = mask.sum().item()
    target_active = (target_param != 0.0).sum().item()
    to_drop = target_active - active
    if to_drop > 0:
        abs_weights = torch.abs(target_param)
        abs_weights[abs_weights == 0.0] = float('inf')
        sorted_weights, _ = torch.sort(abs_weights.view(-1))
        threshold = sorted_weights[to_drop]
        target_param[abs_weights < threshold] = 0.0


def log_sparsity_info(masks, step):
    """Logs sparsity info to wandb.

    Args:
        masks: list of dicts, each dict contains 'weight' and 'mask' and 'name' keys
        step: The current training step.
    """
    for mask_dict in masks:
        name = mask_dict['name']
        weight = mask_dict['weight']
        mask = mask_dict['mask']
        num_active = mask.sum().item()
        num_total = weight.numel()
        wandb.log({f'sparsity/sparsity_weight_{name}': (weight == 0).sum().item() / num_total,
                   f'sparsity/sparsity_mask_{name}': (mask == 0).sum().item() / num_total,
                   f'sparsity/num_active_{name}': num_active,
                   f'sparsity/num_total_{name}': num_total}, step=step)
