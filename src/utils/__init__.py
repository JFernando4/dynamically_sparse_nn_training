# from .sparse_network_utilities import get_mask_from_sparse_module, get_dense_weights_from_sparse_module, \
#     get_sparse_mask_using_weight_magnitude, copy_bias_and_weights_to_sparse_module, copy_weights_to_sparse_module, \
#     convert_indices, refill_indices
from .regularization_funcs import apply_regularization_to_sequential_net, apply_regularization_to_net, \
    apply_regularization_to_parameter_list
from .evaluation_functions import compute_accuracy_from_batch, compute_average_gradient_magnitude, set_random_seed, \
    compute_matrix_rank_summaries, compute_average_weight_magnitude, bootstrapped_return
# from .wandb_setup import initialize_wandb, get_wandb_id
from .data_management import get_cifar_data, subsample_cifar_data_set
from .experiment_utils import parse_terminal_arguments, parse_plots_and_analysis_terminal_arguments
from .cifar100_experiment_utils import *
from .permuted_mnist_experiment_utils import compute_dead_units_proportion
from .analysis_and_plotting import COLOR_DICT, aggregate_over_bins, plot_results, plot_avg_with_shaded_region
