# built-in libraries
import time
import os
import pickle
from copy import deepcopy

# third party libraries
import torch
from torch.utils.data import DataLoader
import numpy as np
from torchvision.models.vision_transformer import VisionTransformer

# from ml project manager
from mlproj_manager.problems import CifarDataSet
from mlproj_manager.experiments import Experiment
from mlproj_manager.file_management import store_object_with_several_attempts
from mlproj_manager.util import turn_off_debugging_processes, get_random_seeds, access_dict

from src import initialize_vit, initialize_vit_heads, init_vit_weight_masks
from src.sparsity_functions import set_up_dst_update_function, apply_weight_masks
from src.utils import get_cifar_data


class IncrementalCIFARExperiment(Experiment):

    def __init__(self, exp_params: dict, results_dir: str, run_index: int, verbose=True):
        super().__init__(exp_params, results_dir, run_index, verbose)

        # set debugging options for pytorch
        debug = access_dict(exp_params, key="debug", default=True, val_type=bool)
        turn_off_debugging_processes(debug)
        # define torch device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        """ For reproducibility """
        self.random_seed = get_random_seeds()[self.run_index]
        torch.random.manual_seed(self.random_seed)
        torch.cuda.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        """ Experiment parameters """
        self.data_path = exp_params["data_path"]

        # optimization parameters
        self.stepsize = exp_params["stepsize"]
        self.weight_decay = exp_params["weight_decay"]
        self.momentum = exp_params["momentum"]
        self.use_lr_schedule = access_dict(exp_params, "use_lr_schedule", default=True, val_type=bool)
        self.dropout_prob = access_dict(exp_params, "dropout_prob", default=0.05, val_type=float)

        # dynamic sparse learning parameters
        self.topology_update_freq = access_dict(exp_params, "topology_update_freq", default=0, val_type=int)
        self.epoch_freq = access_dict(exp_params, "epoch_freq", default=False, val_type=bool)
        self.sparsity = access_dict(exp_params, "sparsity", default=0.0, val_type=float)
        dst_methods_names = ["none", "set", "set_r", "set_rf", "rigl", "rigl_r", "rigl_rf", "set_rth", "set_rth_min", "set_ds"]
        self.dst_method = access_dict(exp_params, "dst_method", default="none", val_type=str, choices=dst_methods_names)
        self.drop_fraction = access_dict(exp_params, "drop_fraction", default=0.0, val_type=float)
        self.df_decay = access_dict(exp_params, "df_decay", default=1.0, val_type=float)
        self.use_dst = self.dst_method != "none"
        self.use_set_rth = "rth" in self.dst_method
        self.use_set_ds = "set_ds" in self.dst_method
        self.dst_update_function = set_up_dst_update_function(self.dst_method, init_type="xavier_uniform")
        self.current_df_decay = 1.0
        self.previously_added_masks = None
        self.current_topology_update = 0

        # network resetting parameters
        self.reset_head = access_dict(exp_params, "reset_head", default=False, val_type=bool)
        self.reset_network = access_dict(exp_params, "reset_network", default=False, val_type=bool)

        # problem definition parameters
        self.num_epochs = access_dict(exp_params, "num_epochs", default=1, val_type=int)
        self.current_num_classes = access_dict(exp_params, "initial_num_classes", default=2, val_type=int)
        self.fixed_classes = access_dict(exp_params, "fixed_classes", default=True, val_type=bool)
        self.use_best_network = access_dict(exp_params, "use_best_network", default=False, val_type=bool)

        # shrink and perturb parameters
        self.noise_std = access_dict(exp_params, "noise_std", default=0.0, val_type=float)
        self.perturb_weights_indicator = self.noise_std > 0.0

        """ Training constants """
        self.batch_sizes = {"train": 90, "test": 100, "validation":50}
        self.num_classes = 100
        self.image_dims = (32, 32, 3)
        self.num_images_per_epoch = 50000
        self.num_images_per_class = 450
        self.num_workers = 1 if self.device.type == "cpu" else 12       # for the data loader

        """ Network set up """
        # initialize network
        self.net = VisionTransformer(
            image_size=32,
            patch_size=4,
            num_layers=8,
            num_heads=12,
            hidden_dim=384,
            mlp_dim=1536,
            num_classes=self.num_classes,
            dropout=self.dropout_prob,
            attention_dropout=self.dropout_prob)
        initialize_vit(self.net)
        self.net.to(self.device)

        # initialize masks
        self.net_masks = None
        if self.use_dst:
            self.net_masks = init_vit_weight_masks(self.net, self.sparsity, include_pos_embedding=True)
            apply_weight_masks(self.net_masks)

        # initialize optimizer and loss function
        self.optim = torch.optim.SGD(self.net.parameters(), lr=self.stepsize, momentum=self.momentum,
                                     weight_decay=self.weight_decay)
        self.lr_scheduler = None
        self.loss = torch.nn.CrossEntropyLoss(reduction="mean")

        # initialize training counters
        self.current_epoch = 0
        self.current_minibatch = 0

        """ For data partitioning """
        self.class_increase = 5
        self.class_increase_frequency = 100
        self.all_classes = np.random.permutation(self.num_classes)  # define order classes
        self.best_accuracy = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        self.best_accuracy_model_parameters = {}
        self.best_accuracy_masks = []

        """ For creating experiment checkpoints """
        self.experiment_checkpoints_dir_path = os.path.join(self.results_dir, "experiment_checkpoints")
        self.checkpoint_identifier_name = "current_epoch"
        self.checkpoint_save_frequency = self.class_increase_frequency  # save every time a new class is added
        self.delete_old_checkpoints = True

        """ For summaries """
        self.running_avg_window = 25
        self.current_running_avg_step, self.running_loss, self.running_accuracy = (0, 0.0, 0.0)
        self._initialize_summaries()

    # ------------------------------ Methods for initializing the experiment ------------------------------
    def _initialize_summaries(self):
        """
        Initializes the summaries for the experiment
        """
        if self.fixed_classes:
            num_images_per_epoch = self.num_images_per_class * self.num_classes
            total_checkpoints = (num_images_per_epoch * self.num_epochs) // (self.running_avg_window * self.batch_sizes["train"])
        else:
            number_of_tasks = np.arange(self.num_epochs // self.class_increase_frequency) + 1
            number_of_image_per_task = self.num_images_per_class * self.class_increase
            bin_size = (self.running_avg_window * self.batch_sizes["train"])
            total_checkpoints = np.sum(number_of_tasks * self.class_increase_frequency * number_of_image_per_task // bin_size)

        train_prototype_array = torch.zeros(total_checkpoints, device=self.device, dtype=torch.float32)
        self.results_dict["train_loss_per_checkpoint"] = torch.zeros_like(train_prototype_array)
        self.results_dict["train_accuracy_per_checkpoint"] = torch.zeros_like(train_prototype_array)

        prototype_array = torch.zeros(self.num_epochs, device=self.device, dtype=torch.float32)
        self.results_dict["epoch_runtime"] = torch.zeros_like(prototype_array)
        # test and validation summaries
        for set_type in ["test", "validation"]:
            self.results_dict[set_type + "_loss_per_epoch"] = torch.zeros_like(prototype_array)
            self.results_dict[set_type + "_accuracy_per_epoch"] = torch.zeros_like(prototype_array)
            self.results_dict[set_type + "_evaluation_runtime"] = torch.zeros_like(prototype_array)
        self.results_dict["class_order"] = self.all_classes

        # dst masks summaries
        if self.use_dst:
            if self.epoch_freq:
                tensor_size = self.num_epochs // self.topology_update_freq
            else:
                tensor_size = total_checkpoints * self.running_avg_window // self.topology_update_freq
            self.results_dict["prop_added_then_removed"] = torch.zeros(tensor_size, device=self.device, dtype=torch.float32)
            if self.dst_method == "set_rth" or self.dst_method == "set_ds":
                self.results_dict["total_removed_per_update"] = torch.zeros(tensor_size, device=self.device, dtype=torch.float32)

    # ----------------------------- For saving and loading experiment checkpoints ----------------------------- #
    def get_experiment_checkpoint(self):
        """ Creates a dictionary with all the necessary information to pause and resume the experiment """

        partial_results = {}
        for k, v in self.results_dict.items():
            partial_results[k] = v if not isinstance(v, torch.Tensor) else v.cpu()

        checkpoint = {
            "model_weights": self.net.state_dict(),
            "net_masks": [m["mask"] for m in self.net_masks],
            "optim_state": self.optim.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(),
            "epoch_number": self.current_epoch,
            "minibatch_number": self.current_minibatch,
            "current_num_classes": self.current_num_classes,
            "all_classes": self.all_classes,
            "current_running_avg_step": self.current_running_avg_step,
            "partial_results": partial_results
        }

        return checkpoint

    def load_checkpoint_data_and_update_experiment_variables(self, file_path):
        """
        Loads the checkpoint and assigns the experiment variables the recovered values
        :param file_path: path to the experiment checkpoint
        :return: (bool) if the variables were succesfully loaded
        """

        with open(file_path, mode="rb") as experiment_checkpoint_file:
            checkpoint = pickle.load(experiment_checkpoint_file)

        self.net.load_state_dict(checkpoint["model_weights"])
        self.optim.load_state_dict(checkpoint["optim_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        torch.cuda.set_rng_state(checkpoint["cuda_rng_state"])
        np.random.set_state(checkpoint["numpy_rng_state"])
        self.current_epoch = checkpoint["epoch_number"]
        self.current_minibatch = checkpoint["minibatch_number"]
        self.current_num_classes = checkpoint["current_num_classes"]
        self.all_classes = checkpoint["all_classes"]
        self.current_running_avg_step = checkpoint["current_running_avg_step"]

        partial_results = checkpoint["partial_results"]
        for k, v in self.results_dict.items():
            self.results_dict[k] = partial_results[k] if not isinstance(partial_results[k], torch.Tensor) else partial_results[k].to(self.device)

        if not self.use_dst:
            return

        for i, mask in enumerate(self.net_masks):
            self.net_masks[i]["mask"] = checkpoint["net_masks"][i].to(self.device)

    # --------------------------------------- For storing summaries --------------------------------------- #
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
            loss, accuracy = self.evaluate_network(data_loader)
            evaluation_time = time.perf_counter() - evaluation_start_time

            if compare_to_best:
                if accuracy > self.best_accuracy:
                    self.best_accuracy = accuracy
                    self.best_accuracy_model_parameters = deepcopy(self.net.state_dict())
                    self.best_accuracy_masks = [deepcopy(m["mask"]) for m in self.net_masks]

            # store summaries
            self.results_dict[data_name + "_evaluation_runtime"][epoch_number] += torch.tensor(evaluation_time, dtype=torch.float32)
            self.results_dict[data_name + "_loss_per_epoch"][epoch_number] += loss
            self.results_dict[data_name + "_accuracy_per_epoch"][epoch_number] += accuracy

            # print progress
            self._print("\t\t{0} accuracy: {1:.4f}".format(data_name, accuracy))

        self.net.train()
        self._print("\t\tEpoch run time in seconds: {0:.4f}".format(epoch_runtime))

    def evaluate_network(self, test_data: DataLoader):
        """
        Evaluates the network on the test data
        :param test_data: a pytorch DataLoader object
        :return: (torch.Tensor) test loss, (torch.Tensor) test accuracy
        """

        avg_loss = 0.0
        avg_acc = 0.0
        num_test_batches = 0
        with torch.no_grad():
            for _, sample in enumerate(test_data):
                images = sample["image"].to(self.device)
                test_labels = sample["label"].to(self.device)
                test_predictions = self.net.forward(images)[:, self.all_classes[:self.current_num_classes]]

                avg_loss += self.loss(test_predictions, test_labels)
                avg_acc += torch.mean((test_predictions.argmax(axis=1) == test_labels.argmax(axis=1)).to(torch.float32))
                num_test_batches += 1

        return avg_loss / num_test_batches, avg_acc / num_test_batches

    # ------------------------------------- For running the experiment ------------------------------------- #
    def run(self):
        # load data
        training_data, training_dl = get_cifar_data(self.data_path, train=True, validation=False,
                                                    batch_size=self.batch_sizes["train"], num_workers=self.num_workers)
        val_data, val_dl = get_cifar_data(self.data_path, train=True, validation=True,
                                          batch_size=self.batch_sizes["validation"], num_workers=self.num_workers)
        test_data, test_dl = get_cifar_data(self.data_path, train=False, batch_size=self.batch_sizes["test"],
                                            num_workers=self.num_workers)
        # load checkpoint if available
        self.load_experiment_checkpoint()
        # train network
        self.train(train_dataloader=training_dl, test_dataloader=test_dl, val_dataloader=val_dl,
                   test_data=test_data, training_data=training_data, val_data=val_data)
        # if using mlproj_manager, summaries are stored in memory by calling exp.store_results()

    def train(self, train_dataloader: DataLoader, test_dataloader: DataLoader, val_dataloader: DataLoader,
              test_data: CifarDataSet, training_data: CifarDataSet, val_data: CifarDataSet):
        # partition data
        training_data.select_new_partition(self.all_classes[:self.current_num_classes])
        test_data.select_new_partition(self.all_classes[:self.current_num_classes])
        val_data.select_new_partition(self.all_classes[:self.current_num_classes])
        # get lr scheduler and save model parameters
        if self.use_lr_schedule:
            self.lr_scheduler = self.get_lr_scheduler(steps_per_epoch=len(train_dataloader))
        self._save_model_parameters()

        # start training
        for e in range(self.current_epoch, self.num_epochs):
            self._print("\tEpoch number: {0}".format(e + 1))

            epoch_start = time.perf_counter()
            for step_number, sample in enumerate(train_dataloader):
                # sample observationa and target
                image = sample["image"].to(self.device)
                label = sample["label"].to(self.device)

                # reset gradients
                for param in self.net.parameters(): param.grad = None   # apparently faster than optim.zero_grad()

                # compute prediction and loss
                predictions = self.net.forward(image)[:, self.all_classes[:self.current_num_classes]]
                current_loss = self.loss(predictions, label)
                detached_loss = current_loss.detach().clone()

                # backpropagate and update weights
                current_loss.backward()
                self.optim.step()
                self.inject_noise()
                if self.use_lr_schedule:
                    self.lr_scheduler.step()
                if self.use_dst:
                    apply_weight_masks(self.net_masks)

                # store summaries
                current_accuracy = torch.mean((predictions.argmax(axis=1) == label.argmax(axis=1)).to(torch.float32))
                self.running_loss += detached_loss
                self.running_accuracy += current_accuracy.detach()
                if (step_number + 1) % self.running_avg_window == 0:
                    self._print("\t\tStep Number: {0}".format(step_number + 1))
                    self._store_training_summaries()

                self.current_minibatch += 1
                if self.time_to_update_topology(minibatch_loop=True):
                    self.update_topology()
            epoch_end = time.perf_counter()

            self._store_test_summaries(test_dataloader, val_dataloader, epoch_number=e, epoch_runtime=epoch_end - epoch_start)
            self.current_epoch += 1
            if self.time_to_update_topology(minibatch_loop=False):
                self.update_topology()
            self.extend_classes(training_data, test_data, val_data, train_dataloader)

            if self.current_epoch % self.checkpoint_save_frequency == 0:
                self.save_experiment_checkpoint()

    def get_lr_scheduler(self, steps_per_epoch: int):
        scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optim, max_lr=self.stepsize, anneal_strategy="linear",
                                                        epochs=self.class_increase_frequency,
                                                        steps_per_epoch=steps_per_epoch)
        return scheduler

    def inject_noise(self):
        """ Adds a small amount of random noise to the parameters of the network """
        if not self.perturb_weights_indicator: return

        with torch.no_grad():
            for param in self.net.parameters():
                param.add_(torch.randn(param.size(), device=param.device) * self.noise_std)

    def time_to_update_topology(self, minibatch_loop: bool = True):
        if not self.use_dst:
            return False
        if minibatch_loop and self.epoch_freq:
            return False
        if not minibatch_loop and not self.epoch_freq:
            return False
        if minibatch_loop:
            return (self.current_minibatch % self.topology_update_freq) == 0
        return (self.current_epoch % self.topology_update_freq) == 0

    def update_topology(self):
        """
        Updates the neural network topology according to the chosen dst algorithm
        """
        removed_masks = []
        added_masks = []
        for mask in self.net_masks:
            if self.use_set_rth:
                third_arg = (self.df_decay * mask["init_std"], )
            elif self.use_set_ds:
                third_arg = (mask["init_func"], self.df_decay)
            else:
                third_arg = (int(self.current_df_decay * self.drop_fraction * mask["mask"].sum()), )

            old_mask = deepcopy(mask["mask"])
            new_mask = self.dst_update_function(mask["mask"], mask["weight"], *third_arg)
            mask_difference = old_mask - new_mask
            added_masks.append(torch.clip(mask_difference, min=-1.0, max=0.0).abs())
            removed_masks.append(torch.clip(mask_difference, min=0.0, max=1.0))
            mask["mask"] = new_mask

        self.current_df_decay = self.df_decay * self.current_df_decay if self.df_decay < 1.0 else 1.0
        self.store_mask_update_summary(removed_masks, added_masks)
        self.current_topology_update += 1

    def store_mask_update_summary(self, removed_masks: list, added_masks: list):
        """
        Computes and storesthe proportion of weights that were removed in the current topology update that were just
        added in the previous topology update

        Args:
            removed_masks: list of masks for weights that were removed in each layer for the current topology update
            added_masks: list of masks for weights that were added in each layer in the current topology update

        Return:
            None, but updates self.results_dict and self.previously_added_masks
        """

        if self.previously_added_masks is not None:
            total_removed = 0
            total_added_then_removed = 0
            for prev_added, recently_removed in zip(self.previously_added_masks, removed_masks):
                total_removed += recently_removed.sum()
                total_added_then_removed += ((prev_added + recently_removed) == 2.0).sum()
            if total_removed == 0:
                prop_added_then_removed = 0.0
            else:
                prop_added_then_removed = total_added_then_removed / total_removed
            # print("Total removed: {0}, decay factor: {1}".format(total_removed, self.current_df_decay))
            # print("Proportion of added then removed: {0:.4f}".format(prop_added_then_removed))
            self.results_dict["prop_added_then_removed"][self.current_topology_update] += prop_added_then_removed
            if self.dst_method == "set_rth" or self.dst_method == "set_ds":
                self.results_dict["total_removed_per_update"][self.current_topology_update] += total_removed

        self.previously_added_masks = added_masks

    def extend_classes(self, training_data: CifarDataSet, test_data: CifarDataSet, val_data: CifarDataSet,
                       train_dataloader: DataLoader):
        """
        Adds 5 new classes to the data set with certain frequency
        """
        if (self.current_epoch % self.class_increase_frequency) == 0 and (not self.fixed_classes):
            self._print("Best accuracy in the task: {0:.4f}".format(self.best_accuracy))
            if self.use_best_network:
                self.net.load_state_dict(self.best_accuracy_model_parameters)
                for mask_dict, best_mask in zip(self.net_masks, self.best_accuracy_masks):
                    mask_dict["mask"] = best_mask
            self.best_accuracy = torch.zeros_like(self.best_accuracy)
            self.best_accuracy_model_parameters = {}
            self.best_accuracy_masks = []
            self._save_model_parameters()
            self.current_df_decay = 1.0

            if self.current_num_classes == self.num_classes: return

            self.current_num_classes += self.class_increase
            training_data.select_new_partition(self.all_classes[:self.current_num_classes])
            test_data.select_new_partition(self.all_classes[:self.current_num_classes])
            val_data.select_new_partition(self.all_classes[:self.current_num_classes])

            self._print("\tNew class added...")
            if self.reset_head:
                initialize_vit_heads(self.net.heads)
            if self.reset_network:
                initialize_vit(self.net)
                self.optim = torch.optim.SGD(self.net.parameters(), lr=self.stepsize, momentum=self.momentum,
                                             weight_decay=self.weight_decay)
            if self.use_lr_schedule:
                self.lr_scheduler = self.get_lr_scheduler(steps_per_epoch=len(train_dataloader))

    def _save_model_parameters(self):
        """ Stores the parameters of the model, so it can be evaluated after the experiment is over """

        model_parameters_dir_path = os.path.join(self.results_dir, "model_parameters")
        masks = None if not self.use_dst else [m["mask"] for m in self.net_masks]
        os.makedirs(model_parameters_dir_path, exist_ok=True)

        file_name = "index-{0}_epoch-{1}.pt".format(self.run_index, self.current_epoch)
        file_path = os.path.join(model_parameters_dir_path, file_name)

        store_object_with_several_attempts((self.net.state_dict(), masks), file_path, storing_format="torch",
                                           num_attempts=10)


def parse_terminal_arguments():
    """ Reads experiment arguments """
    import argparse
    argument_parser = argparse.ArgumentParser()

    # default values for stepsize, weight_decay, and dropout_prob found in a parameter sweep with lr scheduler
    argument_parser.add_argument("--config_file", action="store", type=str, required=True,
                                 help="JSON file with experiment parameters.")
    argument_parser.add_argument("--run_index", action="store", type=int, default=0,
                                 help="This determines the random seed for the experiment.")
    argument_parser.add_argument("--verbose", action="store_true", default=False)

    return argument_parser.parse_args()


def main():
    """
    This is a quick demonstration of how to run the experiments. For a more systematic run, use the mlproj_manager
    scheduler.
    """
    from mlproj_manager.file_management.file_and_directory_management import read_json_file
    terminal_arguments = parse_terminal_arguments()
    experiment_parameters = read_json_file(terminal_arguments.config_file)
    file_path = os.path.dirname(os.path.abspath(__file__))

    experiment_parameters["data_path"] = os.path.join(file_path, "data")
    print(experiment_parameters)
    relevant_parameters = experiment_parameters["relevant_parameters"]
    results_dir_name = "{0}-{1}".format(relevant_parameters[0], experiment_parameters[relevant_parameters[0]])
    for relevant_param in relevant_parameters[1:]:
        results_dir_name += "_" + relevant_param + "-" + str(experiment_parameters[relevant_param])

    initial_time = time.perf_counter()
    exp = IncrementalCIFARExperiment(experiment_parameters,
                                     results_dir=os.path.join(file_path, "results", results_dir_name),
                                     run_index=terminal_arguments.run_index,
                                     verbose=terminal_arguments.verbose)
    exp.run()
    exp.store_results()
    final_time = time.perf_counter()
    print("The running time in minutes is: {0:.2f}".format((final_time - initial_time) / 60))


if __name__ == "__main__":
    main()
