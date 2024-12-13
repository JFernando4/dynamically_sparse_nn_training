import torch
import numpy as np

from mlproj_manager.util import get_random_seeds
from scipy.linalg import svd
from scipy.stats import bootstrap
from tqdm import tqdm


def set_random_seed(seed_index: int):
    """ Sets the random seed of torch, cuda, and numpy """
    random_seed = get_random_seeds()[seed_index]    # this function produces always the same random integers
    torch.random.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)


@torch.no_grad()
def compute_accuracy_from_batch(predictions: torch.Tensor, labels: torch.Tensor):
    """
    Computes accuracy based on a batch of predictions and labels

    Args:
        predictions: tensor of shape (n, c) where n is the batch size and c is the number of classes
        labels: tensor of same shape as predictions

    Return:
        Scalar tensor corresponding to the accuracy, a float between 0 and 1
    """
    return torch.mean((predictions.argmax(axis=1) == labels.argmax(axis=1)).to(torch.float32))

def compute_average_gradient_magnitude(model: torch.nn.Module) -> float:
    """
    computes the average gradient magnitude of a network
    """
    grad_magnitude_summ = 0.0
    total_params = 0.0

    for p in model.parameters():
        if p.grad is not None:
            grad_magnitude_summ += p.grad.abs().sum()
            total_params += p.numel()

    return float(grad_magnitude_summ / total_params)


def compute_matrix_rank_summaries(m: torch.Tensor, prop=0.99, use_scipy=False):
    """
    Computes the rank, effective rank, and approximate rank of a matrix
    Refer to the corresponding functions for their definitions
    :param m: (float np array) a rectangular matrix
    :param prop: (float) proportion used for computing the approximate rank
    :param use_scipy: (bool) indicates whether to compute the singular values in the cpu, only matters when using
                                  a gpu
    :return: (torch int32) rank, (torch float32) effective rank, (torch int32) approximate rank
    """
    if use_scipy:
        np_m = m.cpu().numpy()
        sv = torch.tensor(svd(np_m, compute_uv=False, lapack_driver="gesvd"), device=m.device)
    else:
        sv = torch.linalg.svdvals(m)    # for large matrices, svdvals may fail to converge in gpu, but not cpu
    rank = torch.count_nonzero(sv).to(torch.int32)
    effective_rank = compute_effective_rank(sv)
    approximate_rank = compute_approximate_rank(sv, prop=prop)
    approximate_rank_abs = compute_abs_approximate_rank(sv, prop=prop)
    return rank, effective_rank, approximate_rank, approximate_rank_abs


def compute_effective_rank(sv: torch.Tensor):
    """
    Computes the effective rank as defined in this paper: https://ieeexplore.ieee.org/document/7098875/
    When computing the shannon entropy, 0 * log 0 is defined as 0
    :param sv: (float torch Tensor) an array of singular values
    :return: (float torch Tensor) the effective rank
    """
    norm_sv = sv / torch.sum(torch.abs(sv))
    entropy = torch.tensor(0.0, dtype=torch.float32, device=sv.device)
    for p in norm_sv:
        if p > 0.0:
            entropy -= p * torch.log(p)

    effective_rank = torch.tensor(np.e) ** entropy
    return effective_rank.to(torch.float32)


def compute_approximate_rank(sv: torch.Tensor, prop=0.99):
    """
    Computes the approximate rank as defined in this paper: https://arxiv.org/pdf/1909.12255.pdf
    :param sv: (float np array) an array of singular values
    :param prop: (float) proportion of the variance captured by the approximate rank
    :return: (torch int 32) approximate rank
    """
    sqrd_sv = sv ** 2
    normed_sqrd_sv = torch.flip(torch.sort(sqrd_sv / torch.sum(sqrd_sv))[0], dims=(0,))   # descending order
    cumulative_ns_sv_sum = 0.0
    approximate_rank = 0
    while cumulative_ns_sv_sum < prop:
        cumulative_ns_sv_sum += normed_sqrd_sv[approximate_rank]
        approximate_rank += 1
    return torch.tensor(approximate_rank, dtype=torch.int32)


def compute_abs_approximate_rank(sv: torch.Tensor, prop=0.99):
    """
    Computes the approximate rank as defined in this paper, just that we won't be squaring the singular values
    https://arxiv.org/pdf/1909.12255.pdf
    :param sv: (float np array) an array of singular values
    :param prop: (float) proportion of the variance captured by the approximate rank
    :return: (torch int 32) approximate rank
    """
    sqrd_sv = sv
    normed_sqrd_sv = torch.flip(torch.sort(sqrd_sv / torch.sum(sqrd_sv))[0], dims=(0,))   # descending order
    cumulative_ns_sv_sum = 0.0
    approximate_rank = 0
    while cumulative_ns_sv_sum < prop:
        cumulative_ns_sv_sum += normed_sqrd_sv[approximate_rank]
        approximate_rank += 1
    return torch.tensor(approximate_rank, dtype=torch.int32)


@torch.no_grad()
def compute_average_weight_magnitude(net: torch.nn.Module):
    """ computes the average weight magnitude of the network """

    weight_magnitude = 0.0
    total_weights = 0.0
    ln_weight_magnitude = 0.0
    ln_total_weights = 0.0

    for n, p in net.named_parameters():
        if p.requires_grad:
            weight_magnitude += p.abs().sum()
            total_weights += p.numel()
            if ("weight" in n) and ("ln" in n):
                ln_weight_magnitude += p.abs().sum()
                ln_total_weights += p.numel()

    average_weight_magnitude = weight_magnitude / total_weights
    average_ln_weight_magnitude = 0.0 if ln_total_weights == 0.0 else ln_weight_magnitude / ln_total_weights
    return average_weight_magnitude, average_ln_weight_magnitude


def bootstrapped_return(episode_length: np.ndarray, episodic_return: np.ndarray, bin_size: int,
                        total_steps: int, confidence_level: float = 0.95, to_bootstrap: bool = True):
    assert len(episode_length) == len(episodic_return)
    num_runs = len(episode_length)
    avg_ret = np.zeros(total_steps // bin_size)
    steps = np.arange(bin_size, total_steps + bin_size, bin_size)
    min_rets, max_rets = np.zeros(total_steps // bin_size), np.zeros(total_steps // bin_size)
    boot_strapped_ret_low, boot_strapped_ret_high = np.zeros(total_steps // bin_size), np.zeros(total_steps // bin_size)
    for i in tqdm(range(0, total_steps // bin_size)):
        rets = []
        for run in range(num_runs):
            temp_episode_length = episode_length[run][:np.searchsorted(episode_length[run], total_steps) + 1]
            temp_sum_of_rewards = episodic_return[run][:temp_episode_length.shape[0]]
            rets.append(temp_sum_of_rewards[np.logical_and(i * bin_size < temp_episode_length, temp_episode_length <= (i + 1) * bin_size)].mean())
        rets = np.array([rets])
        avg_ret[i] = rets.mean()
        min_rets[i], max_rets[i] = rets.min(), rets.max()
        bos = bootstrap(data=(rets[0, :],), statistic=np.mean, confidence_level=confidence_level)
        boot_strapped_ret_low[i], boot_strapped_ret_high[i] = bos.confidence_interval.low, bos.confidence_interval.high
    return steps, avg_ret, min_rets, max_rets, boot_strapped_ret_low, boot_strapped_ret_high