
# built in libraries
import os

import numpy as np
# third party libraries
import torch
from torch.utils.data import DataLoader
import numpy
# my own packages
from mlproj_manager.file_management import store_object_with_several_attempts
# project source code
from src.utils import compute_accuracy_from_batch


def save_model_parameters(results_dir: str, run_index: int, current_epoch: int, net: torch.nn.Module):
    """ Stores the parameters of the model, so it can be evaluated after the experiment is over """

    model_parameters_dir_path = os.path.join(results_dir, "model_parameters")
    os.makedirs(model_parameters_dir_path, exist_ok=True)

    file_name = "index-{0}_epoch-{1}.pt".format(run_index, current_epoch)
    file_path = os.path.join(model_parameters_dir_path, file_name)

    store_object_with_several_attempts(net.state_dict(), file_path, storing_format="torch", num_attempts=10)


@torch.no_grad()
def evaluate_network(test_data: DataLoader,
                     device: torch.device,
                     loss: torch.nn.Module,
                     net: torch.nn.Module,
                     all_classes: np.ndarray,
                     current_num_classes: int) -> tuple[torch.Tensor, torch.Tensor]:
    """ Evaluates the network on the test data """

    avg_loss = torch.tensor(0.0, device=device)
    avg_acc = torch.tensor(0.0, device=device)
    num_test_batches = 0

    for _, sample in enumerate(test_data):
        images = sample["image"].to(device)
        test_labels = sample["label"].to(device)
        test_predictions = net.forward(images)[:, all_classes[:current_num_classes]]

        avg_loss += loss(test_predictions, test_labels)
        avg_acc += compute_accuracy_from_batch(test_predictions, test_labels)
        num_test_batches += 1

    return avg_loss / num_test_batches, avg_acc / num_test_batches
