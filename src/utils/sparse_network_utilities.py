import torch
from sparselinear import SparseLinear


def get_mask_from_sparse_module(mod: SparseLinear):
    """
    Returns a 2D torch tensor with zeros and ones, where the ones correspond to the indices of the sparse layer
    :param mod: a SparseLinear module
    :return: mask of zeros and ones corresponding to the sparse weight matrix of the module
    """

    mask_indices = mod.weight.indices()
    current_mask = torch.zeros(mod.weight.size(), dtype=torch.float32)
    current_mask[[mask_indices[0], mask_indices[1]]] += 1.0
    return current_mask


def get_dense_weights_from_sparse_module(mod: SparseLinear, mask: torch.Tensor):
    """
    Creates a dense version of the weight matrix of a sparse module where the zero entries are initialize using
    xavier initialization and the non-zero entries are the original weights of the sparse module weight matrix
    :param mod: a SparseLinear module
    :param mask: a mask tensor of the same shape as the dense weight matrix of the SparseLinear layer
    :return: 2D tensor of weights
    """

    mask_indices = mod.weight.indices()
    # initialize weights
    new_weights = torch.zeros(mod.weight.size(), dtype=torch.float32)
    torch.nn.init.xavier_normal_(new_weights)
    # mask out the weights corresponding to the weights of the SparseLinear layer
    negative_mask = 1.0 - mask
    new_weights *= negative_mask
    # insert weights of the SparseLinear layer into the new weight matrix
    new_weights[mask_indices[0], mask_indices[1]] += mod.weight.values()

    return new_weights


def get_sparse_mask_using_weight_magnitude(weights: torch.Tensor, k: int = None, sparsity_level: float = None,
                                           cutoff: float = None):
    """
    Returns a mask of zeros and ones where only the weights above certain cutoff are assigned a value of one. The cutoff
    depends on the sparsity level.
    :param weights: tensor of weights
    :param k: number of top elements to select from the absolute value of the weights
    :param sparsity_level: proportion of weights to prune out
    :param cutoff: a float corresponding to the minimal weight magnitude value; everything under this cutoff is pruned
    :return: tensor of same shape as weights
    """

    abs_weights = torch.abs(weights)

    if k is not None:
        topk_weights = torch.topk(abs_weights.flatten(), k=k)
        mask = torch.zeros_like(abs_weights).flatten()
        mask[topk_weights.indices] += 1.0
        return mask.reshape(weights.shape)

    elif sparsity_level is not None or cutoff is not None:
        if cutoff is None:
            cutoff = torch.quantile(torch.abs(weights), sparsity_level)
        return (abs_weights > cutoff).to(torch.float32)

    else:
        raise ValueError("Either k, sparsity_level, or cutoff must be given!")


def copy_bias_and_weights_to_sparse_module(mod: SparseLinear, bias: torch.Tensor, weights: torch.Tensor,
                                           mask: torch.Tensor):
    """
    Copies the given weights and bias to a give SparseLinear module
    :param mod: a SparseLinear module
    :param bias: torch Tensor of bias term
    :param weights: torch Tensor of weights
    :param mask: torch Tensor of the same shape as weights
    :return: None, but modifies mod
    """

    with torch.no_grad():
        mod.bias.multiply_(0.0)
        mod.bias.add_(bias)

    copy_weights_to_sparse_module(mod, weights[mask > 0.0])


def copy_weights_to_sparse_module(mod: SparseLinear, new_weights: torch.Tensor):
    """
    Copies the given weights to a given SparseLinear module
    :param mod: a SparseLinear module
    :param new_weights: torch tensor of weights
    :return: None, but modifies the weights in the given module
    """

    with torch.no_grad():
        mod.weights.multiply_(0.0)
        mod.weights.add_(new_weights)


def convert_indices(indices: torch.Tensor, num_cols: int, from_1d_to_2d: bool = False):
    """
    Converts between 1D indices and 2D indices
    :param indices: torch 2D or 1D tensor
    :param num_cols: number of columns in the array
    :param from_1d_to_2d: indicates whether to convert from 1d indices to 2d indices
    :return: converted indices
    """
    if from_1d_to_2d:
        new_indices = torch.zeros((2, indices.size()[0]), dtype=indices.dtype)
        new_indices[0, :] = indices // num_cols
        new_indices[1, :] = indices % num_cols
    else:
        new_indices = indices[0, :] * num_cols + indices[1, :]
    return new_indices


def refill_indices(current_indices: torch.Tensor, matrix_dims: torch.Tensor, total_indices: int):
    """
    Given a 2D tensor of indices with a number of entries less than or equal to total_indices, it adds unique indices
    to the tensor until the tensor has exactly the desired amount of indices (given by total_indices)
    :param current_indices: current tensor of indices, first dim should be of size 2, second dim should be total number
                            of current indices
    :param matrix_dims: tensor with the dimensions of the matrix
    :param total_indices: total number of desired indices
    :return: tensor with the desired amount of indices
    """

    current_total_indices = current_indices.shape[1]
    if current_total_indices == total_indices:   # the tensor already has the desired number of indices
        return current_indices

    n, m = matrix_dims
    num_parameters = n * m

    flat_indices = convert_indices(current_indices, num_cols=m, from_1d_to_2d=False)
    # sanity check:
    # print(torch.all((flat_indices // m) == current_indices[0, :]))
    # print(torch.all((flat_indices % m) == current_indices[1, :]))

    while flat_indices.size()[0] < total_indices:
        temp_num_indices = flat_indices.shape[0]
        temp_indices = torch.zeros(total_indices, dtype=flat_indices.dtype)
        temp_indices[:temp_num_indices] += flat_indices
        missing_indices = total_indices - temp_num_indices
        temp_indices[temp_num_indices:] += torch.randint(low=0, high=num_parameters, size=(missing_indices, ))
        flat_indices = torch.unique(temp_indices, sorted=True)      # this might be a bottleneck

    return flat_indices
