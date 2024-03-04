import torch


@torch.no_grad()
def update_one_weight_mask_set(mask, weight, refresh_num, reinit='zero',):
    """Updates the weight mask of one layer.
    SET algorithm: https://arxiv.org/abs/1707.04780
    (but with standard magnitude pruning)

    Args:
        mask: The weight mask.
        weight: The weights of one layer, corresponding to the mask.
        refresh_num: The number of weights to drop and grow.
        reinit: How to reinitialize the weights that are regrown. Options: 'zero', 'kaiming_normal'
        """
    # mask = prune_magnitude(mask, weight, refresh_num)
    if refresh_num == 0.0:
        return mask
    mask = prune_magnitude_optimized(mask, weight, refresh_num)
    if reinit == "random_fixed":
        mask = grow_random_fixed(mask, weight, refresh_num)
    else:
        mask = grow_random(mask, weight, refresh_num, reinit)
    weight.multiply_(mask)
    return mask


@torch.no_grad()
def update_one_weight_mask_rigl(mask, weight, refresh_num, reinit='zero'):
    """Updates the weight mask of one layer.
    RigL algorithm: https://arxiv.org/abs/1911.11134

    Args:
        mask: The weight mask.
        weight: The weights of one layer, corresponding to the mask.
        refresh_num: The number of weights to drop and grow.
        reinit: How to reinitialize the weights that are regrown. Options: 'zero', 'kaiming_normal'
        """
    if refresh_num == 0.0:
        return mask
    mask = prune_magnitude_optimized(mask, weight, refresh_num)
    if reinit == "random_fixed":
        mask = grow_random_fixed_rigl(mask, weight, refresh_num)
    else:
        mask = grow_rigl(mask, weight, refresh_num, reinit)
    weight.multiply_(mask)
    return mask


@torch.no_grad()
def update_one_weight_mask_set_dense_to_sparse(mask, weight: torch.Tensor, init_function, scale_factor: float = 1.0):
    """ Updates the weight mask of one layer by first filling in the zeros of the weight tensor with random values
        according to the given init function, and then pruning using weight magnitude

        Args:
            mask: The weight mask.
            weight: The weights of one layer, corresponding to the mask.
            init_function: Function for initializing the values of masked out weights
            scale_factor: float to multiply random initial weights by
    """

    # generate random initial weights
    dummy_weight = torch.zeros_like(weight)
    init_function(dummy_weight)
    if scale_factor != 1.0:
        dummy_weight *= scale_factor
    # fill zeros in weight matrix with random initial weights
    zeros_indices = torch.where(mask.flatten() == 0.0)[0]
    weight.view(-1)[zeros_indices] += dummy_weight.view(-1)[zeros_indices]
    # prune weight matrix down
    mask = prune_magnitude_from_dense_weights(weight, zeros_indices.numel())
    weight.multiply_(mask)
    return mask


@torch.no_grad()
def update_one_weight_mask_set_random_with_threshold(mask, weight: torch.Tensor, threshold: float,
                                                     reinit_type: str = "thr"):
    """
    Updates the weight mask of one layer based on a given threshold

        Args:
            mask: The weight mask.
            weight: The weights of one layer, corresponding to the mask.
            threshold: Threshold used for pruning and growing
            reinit_type: str indicating how to reinitialize weights "thr" = +/-threshold, "min" = +/-min active weight
    """
    active_weights_indices = torch.where(mask.flatten() == 1.0)[0]
    abs_active_weights = weight.flatten().abs()[active_weights_indices]
    to_prune = torch.where(abs_active_weights < threshold)[0]
    grown_num = to_prune.numel()
    mask.view(-1)[active_weights_indices[to_prune]] = 0.0

    # randomly add more weights
    reinit_val = threshold if reinit_type == "thr" else None
    mask = grow_random_fixed(mask, weight, grown_num, reinit_val=reinit_val)
    weight.multiply_(mask)
    return mask


def set_up_dst_update_function(dst_method_name: str, init_type: str = "xavier_uniform"):
    """
    Returns a dst update function according to the dst_method name

    Args:
        dst_method_name: string corresponding to a dst method, choices: "set", "set_r" (set with random init weights),
                         "rigl", "rigl_r" (rigl with random init weights), "set_ds" (set dense to sparse), or "none"
        init_type: string indicating the type of initialization, choices: "xavier_uniform", "kaiming_normal", "zero"
    Returns:
        lambda function with the appropriate parameters
    """
    if dst_method_name == "set":
        return lambda m, w, rn: update_one_weight_mask_set(m, w, refresh_num=rn, reinit="zero")
    elif dst_method_name == "set_r":
        return lambda m, w, rn: update_one_weight_mask_set(m, w, refresh_num=rn, reinit=init_type)
    elif dst_method_name == "set_rf":
        return lambda m, w, rn: update_one_weight_mask_set(m, w, refresh_num=rn, reinit="random_fixed")
    elif dst_method_name == "rigl":
        return lambda m, w, rn: update_one_weight_mask_rigl(m, w, refresh_num=rn, reinit="zero")
    elif dst_method_name == "rigl_r":
        return lambda m, w, rn: update_one_weight_mask_rigl(m, w, refresh_num=rn, reinit=init_type)
    elif dst_method_name == "rigl_rf":
        return lambda m, w, rn: update_one_weight_mask_rigl(m, w, refresh_num=rn, reinit="random_fixed")
    elif dst_method_name == "set_rth":
        return lambda m, w, thr: update_one_weight_mask_set_random_with_threshold(m,w, threshold=thr, reinit_type="thr")
    elif dst_method_name == "set_rth_min":
        return lambda m, w, thr: update_one_weight_mask_set_random_with_threshold(m,w, threshold=thr, reinit_type="min")
    elif dst_method_name == "set_ds":
        return update_one_weight_mask_set_dense_to_sparse
    elif dst_method_name == "none":
        return None
    else:
        raise ValueError("Not a valid dst method: {0}".format(dst_method_name))


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


def prune_magnitude_optimized(mask, weight, drop_num):
    """Prunes the weight mask by dropping the smallest magnitude weights."""
    active_indices = torch.where(mask.flatten() == 1.0)[0]
    active_weights = weight.flatten().abs()[active_indices]
    sorted_indices = torch.argsort(active_weights)
    mask.view(-1)[active_indices[sorted_indices[:drop_num]]] = 0.0
    return mask


def prune_magnitude_from_dense_weights(weight, drop_num):
    """ Creates a mask by dropping the weights with the smallest magnitude """

    abs_weight = torch.abs(weight).flatten()
    indices = torch.argsort(abs_weight)
    mask = torch.ones_like(weight, requires_grad=False)
    mask.view(-1)[indices[:drop_num]] = 0.0
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


@torch.no_grad()
def grow_random_fixed_rigl(mask, weight, grow_num):
    """
     Grow connections in the weight mask by selecting the entries where the gradient is largest. The initialization
     value for the new weights is the minimum of the absolute value of the current active weights times the sign of
     the gradient.
    """
    non_active = len(torch.where(mask.flatten() == 0)[0])
    if non_active == 0:
        return mask
    # get minimum of the absolute value of the current active weights
    active_indices = torch.where(mask.flatten() == 1.0)[0]
    min_abs_active_weights = weight.abs().flatten()[active_indices].min()
    # find and grow where the magnitude of the gradient is largest for inactive weights
    grad = torch.abs(weight.grad) + 1e-3    # if weight grads happen to be 0, they will get a small value now
    grad = grad * (mask == 0.0)             # only consider non-active weights
    sorted_grads_indices = torch.argsort(grad.flatten(), descending=True)
    to_grow = min(grow_num, non_active)
    to_grow_indices = sorted_grads_indices[:to_grow]
    mask.view(-1)[to_grow_indices] = 1.0
    # initialize new weights to the min of absolute value of active weights times the sign of the gradient
    weight.view(-1)[to_grow_indices] = min_abs_active_weights * torch.sign(weight.grad.view(-1)[to_grow_indices])

    return mask


@torch.no_grad()
def grow_random_fixed(mask, weight, grow_num, reinit_val: float = None):
    """
    Grow connections in the weight mask by randomly selecting inactive weights.
    The value of those weights is set to the provided reinitialization value or the min of the current active weights.
    """
    # get the minimum of the current active indices
    active_indices = torch.where(mask.flatten() == 1.0)[0]
    if reinit_val is None:
        reinit_val = weight.abs().flatten()[active_indices].min()
    # grow weights
    indices = torch.where(mask.flatten() == 0.0)[0]
    non_active = len(indices)
    if non_active == 0:
        return mask
    shuffle = torch.randperm(non_active)
    indices = indices[shuffle]
    to_grow = min(grow_num, non_active)
    indices = indices[:to_grow]
    mask.view(-1)[indices] = 1
    # assign +/- the min abs value to the new weights
    weight.view(-1)[indices[:len(indices) // 2]] = reinit_val
    weight.view(-1)[indices[len(indices) // 2:]] = -reinit_val
    return mask


def grow_rigl(mask, weight, grow_num, reinit):
    """Grows connections in the weight mask by growing where the gradient is largest."""
    non_active = len(torch.where(mask.flatten() == 0)[0])
    if non_active == 0:
        return mask

    grad = torch.abs(weight.grad) + 1e-3  # if weight grads happen to be 0, they will get a small value now
    grad = grad * (mask == 0).float()  # only consider non-active weights
    indices = torch.argsort(grad.flatten(), descending=True)

    to_grow = min(grow_num, non_active)
    indices = indices[:to_grow]
    mask.view(-1)[indices] = 1.
    if reinit != 'zero':
        reinit_grown_weights(weight, indices, reinit)
    return mask


@torch.no_grad()
def reinit_grown_weights(weight, indices, reinit):
    if reinit == 'kaiming_normal':
        temp_weight = torch.empty_like(weight)
        torch.nn.init.kaiming_normal_(temp_weight)
        weight.view(-1)[indices] = temp_weight.view(-1)[indices]
    elif reinit == "xavier_uniform":
        temp_weight = torch.empty_like(weight)
        torch.nn.init.xavier_uniform_(temp_weight)
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


def init_weight_mask_from_tensor(weight_tensor: torch.Tensor, sparsity):
    """
    Initializes a weight mask for a tensor of parameters.

    Args:
        weight_tensor: The tensor of parameters to initialize the mask for
        sparsity: The sparsity of the weight mask

    Returns:
        A dict containing the mask and the weights of the layer
    """
    num_pruned = int(weight_tensor.numel() * sparsity)
    mask = torch.ones_like(weight_tensor, dtype=torch.float32, requires_grad=False).to(weight_tensor.device)
    mask.view(-1)[torch.randperm(weight_tensor.numel())[:num_pruned]] = 0.
    return {'weight': weight_tensor, 'mask': mask}


@torch.no_grad()
def apply_weight_masks(masks: list):
    """Applies the weight masks to the weights.

    Args:
        masks: list of dicts, each dict contains 'weight' and 'mask' keys
    """
    for mask_dict in masks:
        mask_dict["weight"].multiply_(mask_dict["mask"])
        # mask_dict['weight'].data *= mask_dict['mask']


def update_weight_masks(masks, drop_fraction, reinit='zero', grow_method='set'):
    """Updates all the weight masks.

    Args:
        masks: list of dicts, each dict contains 'weight' and 'mask' keys
        drop_fraction: The fraction of weights to drop and grow.
        reinit: How to reinitialize the weights that are regrown. Options: 'zero', 'kaiming_normal'
        grow_method: The method to use for growing weights. Options: 'set', 'rigl'
    """
    if grow_method == 'set':
        for mask_dict in masks:
            num_active = mask_dict['mask'].sum().item()
            num_drop = int(num_active * drop_fraction)
            mask_dict['mask'] = update_one_weight_mask_set(mask_dict['mask'], mask_dict['weight'], num_drop, reinit)
    elif grow_method == 'rigl':
        for mask_dict in masks:
            num_active = mask_dict['mask'].sum().item()
            num_drop = int(num_active * drop_fraction)
            mask_dict['mask'] = update_one_weight_mask_rigl(mask_dict['mask'], mask_dict['weight'], num_drop, reinit)
    else:
        raise ValueError(f'Unknown DST grow method {grow_method}')
