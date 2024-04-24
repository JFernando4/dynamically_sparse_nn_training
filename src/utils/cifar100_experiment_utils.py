"""
Functions and parent class for running CIFAR-100 experiments.
"""

# built in libraries
import os
import time
from copy import deepcopy

# third party libraries
import numpy as np
import torch
from torch.utils.data import DataLoader
import numpy
# my own packages
from mlproj_manager.experiments import Experiment
from mlproj_manager.file_management import store_object_with_several_attempts
from mlproj_manager.util import turn_off_debugging_processes, get_random_seeds, access_dict
from mlproj_manager.problems import CifarDataSet
# project source code
from src.utils import compute_accuracy_from_batch


class IncrementalCIFARExperiment(Experiment):

    def __init__(self, exp_params: dict, results_dir: str, run_index: int, verbose=True):
        super().__init__(exp_params, results_dir, run_index, verbose)

        # set debugging options for pytorch
        debug = access_dict(exp_params, key="debug", default=True, val_type=bool)
        turn_off_debugging_processes(debug)

        # define torch device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        """ For reproducibility """
        random_seeds = get_random_seeds()
        self.random_seed = random_seeds[self.run_index]
        torch.random.manual_seed(self.random_seed)
        torch.cuda.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        """ Experiment parameters """
        self.data_path = exp_params["data_path"]

        # problem definition parameters
        self.num_epochs = access_dict(exp_params, "num_epochs", default=1, val_type=int)
        self.current_num_classes = access_dict(exp_params, "initial_num_classes", default=2, val_type=int)
        self.fixed_classes = access_dict(exp_params, "fixed_classes", default=True, val_type=bool)
        self.use_best_network = access_dict(exp_params, "use_best_network", default=False, val_type=bool)
        self.compare_loss = access_dict(exp_params, "compare_loss", default=False, val_type=bool)

        """ Training constants """
        self.batch_sizes = {"train": 90, "test": 100, "validation": 50}
        self.num_classes = 100
        self.image_dims = (32, 32, 3)
        self.flat_image_dims = int(np.prod(self.image_dims))
        self.num_images_per_epoch = 50000
        self.num_test_samples = 10000
        self.num_images_per_class = 450
        self.num_workers = 1 if self.device.type == "cpu" else 12  # for the data loader

        # define loss function
        self.loss = torch.nn.CrossEntropyLoss(reduction="mean")

        # move network to device
        self.current_epoch = 0
        self.current_minibatch = 0

        """ For data partitioning """
        self.class_increase = 5
        self.all_classes = np.random.permutation(self.num_classes)
        self.best_accuracy = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        self.best_loss = torch.ones_like(self.best_accuracy) * torch.inf
        self.best_model_parameters = {}

        """ For creating experiment checkpoints """
        self.experiment_checkpoints_dir_path = os.path.join(self.results_dir, "experiment_checkpoints")
        self.checkpoint_identifier_name = "current_epoch"
        self.delete_old_checkpoints = True

        """ For summaries """
        self.running_avg_window = 25
        self.current_running_avg_step, self.running_loss, self.running_accuracy = (0, 0.0, 0.0)
        self.previously_removed_weights = None

        """ Placeholders """
        self.net = torch.nn.Linear(1,1)   # Either ResNet18 or VisionTransformer
        self.optim = torch.optim.Optimizer                      # SGD
        self.checkpoint_save_frequency = -1                     # positive integer
        self.class_increase_frequency = -1                      # positive integer
        self.use_cbpw = None                                    # bool
        self.topology_update_freq = -1                          # non-negative integer
        self.current_topology_update = -1                       # non-negative integer
        self.prune_method = "placeholder"                       # string from ["none", "magnitude", "redo", "gf", "hess_approx"]

    def _initialize_summaries(self):
        """
        Initializes the summaries for the experiment
        """
        array_arguments = {"device": self.device, "dtype": torch.float32}
        if self.fixed_classes:
            num_images_per_epoch = self.num_images_per_class * self.num_classes
            total_checkpoints = (num_images_per_epoch * self.num_epochs) // (self.running_avg_window * self.batch_sizes["train"])
        else:
            number_of_tasks = np.arange(self.num_epochs // self.class_increase_frequency) + 1
            class_increase = 5
            number_of_image_per_task = self.num_images_per_class * class_increase
            bin_size = (self.running_avg_window * self.batch_sizes["train"])
            total_checkpoints = np.sum(number_of_tasks * self.class_increase_frequency * number_of_image_per_task // bin_size)

        self.results_dict["train_loss_per_checkpoint"] = torch.zeros(total_checkpoints, **array_arguments)
        self.results_dict["train_accuracy_per_checkpoint"] = torch.zeros(total_checkpoints, **array_arguments)

        self.results_dict["epoch_runtime"] = torch.zeros(self.num_epochs, **array_arguments)
        # test and validation summaries
        for set_type in ["test", "validation"]:
            self.results_dict[set_type + "_loss_per_epoch"] = torch.zeros(self.num_epochs, **array_arguments)
            self.results_dict[set_type + "_accuracy_per_epoch"] = torch.zeros(self.num_epochs, **array_arguments)
            self.results_dict[set_type + "_evaluation_runtime"] = torch.zeros(self.num_epochs, **array_arguments)
        self.results_dict["class_order"] = self.all_classes

        if self.use_cbpw:
            tensor_size = total_checkpoints * self.running_avg_window // self.topology_update_freq
            self.results_dict["prop_added_then_removed"] = torch.zeros(tensor_size, **array_arguments)
            if "redo" in self.prune_method:
                self.results_dict["total_removed_per_update"] = torch.zeros(tensor_size, **array_arguments)

    def _store_training_summaries(self):
        self.results_dict["train_loss_per_checkpoint"][self.current_running_avg_step] += self.running_loss / self.running_avg_window
        self.results_dict["train_accuracy_per_checkpoint"][self.current_running_avg_step] += self.running_accuracy / self.running_avg_window

        self._print("\t\tOnline accuracy: {0:.2f}".format(self.running_accuracy / self.running_avg_window))
        self.running_loss *= 0.0
        self.running_accuracy *= 0.0
        self.current_running_avg_step += 1

    def _store_test_summaries(self, test_data: DataLoader, val_data: DataLoader, epoch_number: int, epoch_runtime: float):
        """ Computes test summaries and stores them in results dir """

        self.results_dict["epoch_runtime"][epoch_number] += torch.tensor(epoch_runtime, dtype=torch.float32)

        self.net.eval()
        for data_name, data_loader, compare_to_best in [("test", test_data, False), ("validation", val_data, True)]:
            # evaluate on data
            evaluation_start_time = time.perf_counter()
            loss, accuracy = evaluate_network(data_loader, self.device, self.loss, self.net, self.all_classes, self.current_num_classes)
            evaluation_time = time.perf_counter() - evaluation_start_time

            if compare_to_best:
                if accuracy > self.best_accuracy:
                    self.best_accuracy = accuracy
                    if not self.compare_loss:
                        self.best_model_parameters = deepcopy(self.net.state_dict())
                if loss < self.best_loss:
                    self.best_loss = loss
                    if self.compare_loss:
                        self.best_model_parameters = deepcopy(self.net.state_dict())

            # store summaries
            self.results_dict[data_name + "_evaluation_runtime"][epoch_number] += torch.tensor(evaluation_time, dtype=torch.float32)
            self.results_dict[data_name + "_loss_per_epoch"][epoch_number] += loss
            self.results_dict[data_name + "_accuracy_per_epoch"][epoch_number] += accuracy

            # print progress
            self._print("\t\t{0} accuracy: {1:.4f}".format(data_name, accuracy))

        self.net.train()
        self._print("\t\tEpoch run time in seconds: {0:.4f}".format(epoch_runtime))

    def _store_mask_update_summary(self, cbpw_summaries: dict[str, tuple]) -> None:
        """
        Computes and stores the proportion of weights that were removed in the current topology update that were also
        removed in the last one

        Args:
            cbpw_summaries: dict corresponding to the output of the function update_weights in
                            cbpw_functions.weight_matrix_updates.py
        """
        removed_masks = [v[0] for v in cbpw_summaries.values()]
        total_removed = sum([v[1] for v in cbpw_summaries.values()])

        if self.previously_removed_weights is not None:
            total_added_then_removed = 0
            for prev_removed, recently_removed in zip(self.previously_removed_weights, removed_masks):
                total_added_then_removed += ((prev_removed + recently_removed) == 0.0).sum()

            if total_removed == 0:
                prop_added_then_removed = 0.0
            else:
                prop_added_then_removed = total_added_then_removed / total_removed
            # print("Total removed: {0}".format(total_removed))
            # print("Proportion of added then removed: {0:.4f}".format(prop_added_then_removed))
            self.results_dict["prop_added_then_removed"][self.current_topology_update] += prop_added_then_removed
            if "redo" in self.prune_method:
                self.results_dict["total_removed_per_update"][self.current_topology_update] += total_removed

        self.previously_removed_weights = removed_masks

    def extend_classes(self, training_data: CifarDataSet, test_data: CifarDataSet, val_data: CifarDataSet):
        """
        Adds one new class to the data set with certain frequency
        """
        if (self.current_epoch % self.class_increase_frequency) == 0 and (not self.fixed_classes):
            self._print("Best accuracy in the task: {0:.4f}".format(self.best_accuracy))
            if self.use_best_network:
                self.net.load_state_dict(self.best_model_parameters)
            self.best_accuracy = torch.zeros_like(self.best_accuracy)
            self.best_loss = torch.ones_like(self.best_accuracy) * torch.inf
            self.best_model_parameters = {}
            save_model_parameters(self.results_dir, self.run_index, self.current_epoch, self.net)

            if self.current_num_classes == self.num_classes: return

            increase = 5
            self.current_num_classes += increase
            self.partition_data(training_data, test_data, val_data)

            self._print("\tNew class added...")
            self.post_class_increase_processing()
            return True
        return False

    def post_class_increase_processing(self):
        """
        Performs optional operations after the number of classes has been increased, such as resetting the network,
        resetting the head of the network, resetting layer norm parameters, etc
        """
        raise NotImplementedError

    def partition_data(self, training_data: CifarDataSet, test_data: CifarDataSet, val_data: CifarDataSet) -> None:
        """ Partitions data given the current number of classes """
        training_data.select_new_partition(self.all_classes[:self.current_num_classes])
        test_data.select_new_partition(self.all_classes[:self.current_num_classes])
        val_data.select_new_partition(self.all_classes[:self.current_num_classes])


def save_model_parameters(results_dir: str, run_index: int, current_epoch: int, net: torch.nn.Module):
    """ Stores the parameters of the model, so it can be evaluated after the experiment is over """

    model_parameters_dir_path = os.path.join(results_dir, "model_parameters")
    os.makedirs(model_parameters_dir_path, exist_ok=True)

    file_name = "index-{0}_epoch-{1}.pt".format(run_index, current_epoch)
    file_path = os.path.join(model_parameters_dir_path, file_name)

    store_object_with_several_attempts(net.state_dict(), file_path, storing_format="torch", num_attempts=10)


@torch.no_grad()
def evaluate_network(test_data: DataLoader, device: torch.device, loss: torch.nn.Module, net: torch.nn.Module,
                     all_classes: np.ndarray, current_num_classes: int) -> tuple[torch.Tensor, torch.Tensor]:
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

