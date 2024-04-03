# built-in libraries
import time
import os
import argparse
import pickle
from copy import deepcopy

# third party libraries
from torch.utils.data import DataLoader
import numpy as np

# from ml project manager
from mlproj_manager.experiments import Experiment
from mlproj_manager.problems import MnistDataSet
from mlproj_manager.util import access_dict, Permute, get_random_seeds, turn_off_debugging_processes
from mlproj_manager.util.neural_networks import init_weights_kaiming
from mlproj_manager.file_management import store_object_with_several_attempts

# from src
from src.sparsity_functions.sparsity_funcs import *
from src.networks import RegularizedSGD


class PermutedMNISTExperiment(Experiment):

    def __init__(self, exp_params: dict, results_dir: str, run_index: int, verbose=False):
        super().__init__(exp_params, results_dir, run_index, verbose=verbose)

        # set debugging options for pytorch
        debug = access_dict(exp_params, key="debug", default=True, val_type=bool)
        turn_off_debugging_processes(debug)

        # define torch device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        """ For reproducibility """
        actual_seed = int(get_random_seeds()[run_index])
        torch.random.manual_seed(actual_seed)
        np.random.seed(actual_seed)
        if "cuda" in self.device.type:
            torch.cuda.manual_seed(actual_seed)

        """ Experiment parameters """

        # learning parameters
        self.stepsize = exp_params["stepsize"]
        self.l1_factor = access_dict(exp_params, "l1_factor", default=0.0, val_type=float)
        self.l2_factor = access_dict(exp_params, "l2_factor", default=0.0, val_type=float)

        # architecture parameters
        self.num_layers = exp_params["num_layers"]      # number of hidden layers
        self.num_hidden = exp_params["num_hidden"]      # number of hidden units per hidden layer

        # problem parameters
        self.num_permutations = exp_params["num_permutations"]      # number of permutations (1 permutation = 1 epoch)
        self.steps_per_task = access_dict(exp_params, "steps_per_task", default=60000, val_type=int)
        assert self.steps_per_task <= 60000
        self.current_task_steps = 0
        self.current_experiment_step = 0

        # dynamic sparse learning parameters
        self.topology_update_freq = access_dict(exp_params, "topology_update_freq", default=0, val_type=int)
        self.sparsity = access_dict(exp_params, "sparsity", default=0.0, val_type=float)
        self.sparse_network = self.sparsity > 0
        dst_methods_names = ["none", "set", "set_r", "set_rf", "rigl", "rigl_r", "rigl_rf", "set_ds"]
        self.dst_method = access_dict(exp_params, "dst_method", default="none", val_type=str, choices=dst_methods_names)
        self.use_dst = self.dst_method != "none"
        self.use_set_ds = "set_ds" in self.dst_method
        self.dst_update_function = set_up_dst_update_function(self.dst_method, init_type="xavier_uniform")
        self.drop_fraction = access_dict(exp_params, "drop_fraction", default=0.0, val_type=float)
        self.dst_scale = access_dict(exp_params, "dst_scale", default=1.0, val_type=float)
        self.max_steps_for_topology_update = access_dict(exp_params, "max_steps_for_topology_update", default=100, val_type=int)
        self.previously_added_masks = None
        self.current_topology_update = 0

        # paths for loading and storing data
        self.data_path = exp_params["data_path"]
        self.store_parameters = access_dict(exp_params, "store_parameters", default=False, val_type=bool)
        self.parameter_save_frequency = 50  # how often to save the parameters in terms of number of tasks
        self.results_dir = results_dir

        """ Training constants """
        self.batch_size = 1
        self.num_classes = 10
        self.num_inputs = 784
        self.max_num_images_per_permutation = 60000

        """ Network set up """
        self.net, self.masks = self.initialize_network()

        # initialize optimizer
        self.optim = RegularizedSGD(self.net.parameters(),
                                    lr=self.stepsize,
                                    weight_decay=self.l2_factor / self.stepsize,
                                    l1_reg_factor=self.l1_factor / self.stepsize)

        # define loss function
        self.loss = torch.nn.CrossEntropyLoss(reduction="mean")

        # move network to device
        self.net.to(self.device)

        """ Experiment Summaries """
        self.running_avg_window = 100
        self.current_running_avg_step, self.running_loss, self.running_accuracy, self.current_permutation = (0, 0.0, 0.0, 0)
        self.results_dict = {}
        total_ckpts = self.steps_per_task * self.num_permutations // (self.running_avg_window * self.batch_size)
        self.results_dict["train_loss_per_checkpoint"] = torch.zeros(total_ckpts, device=self.device, dtype=torch.float32)
        self.results_dict["train_accuracy_per_checkpoint"] = torch.zeros(total_ckpts, device=self.device, dtype=torch.float32)

        if self.sparse_network:
            total_top_updates = (self.steps_per_task * self.num_permutations) // self.topology_update_freq
            self.results_dict["prop_added_then_removed"] = torch.zeros(total_top_updates, device=self.device, dtype=torch.float32)
            if self.use_set_ds:
                self.results_dict["total_removed_per_update"] = torch.zeros(total_top_updates, device=self.device, dtype=torch.float32)

        """ For creating experiment checkpoints """
        self.current_permutation = 0
        self.experiment_checkpoints_dir_path = os.path.join(self.results_dir, "experiment_checkpoints")
        self.checkpoint_identifier_name = "current_permutation"
        self.checkpoint_save_frequency = 50     # create a checkpoint after this many task changes
        self.load_experiment_checkpoint()

    # ----------------------------- For initializing the experiment ----------------------------- #
    def initialize_network(self):
        """ Initializes the network used for training and the masks of each layer """
        net = torch.nn.Sequential()
        masks = []
        # hidden layers
        in_features = self.num_inputs
        for _ in range(self.num_layers):
            out_features = self.num_hidden
            current_hidden_layer = torch.nn.Linear(in_features, out_features, bias=True)
            net.append(current_hidden_layer)
            if self.sparsity > 0.0:
                masks.append(init_weight_mask(current_hidden_layer, self.sparsity))
                masks[-1]["init_func"] = lambda z: init_weights_kaiming(z, nonlinearity="relu", normal=True)
            net.append(torch.nn.ReLU())
            in_features = out_features
        # output layer, no masks applied to it
        net.append(torch.nn.Linear(self.num_hidden, self.num_classes, bias=True))

        # initialize weights
        net.apply(lambda z: init_weights_kaiming(z, nonlinearity="relu", normal=True))

        return net, masks

    # ----------------------------- For saving and loading experiment checkpoints ----------------------------- #
    def get_experiment_checkpoint(self) -> dict:
        """ Creates a dictionary with all the data necessary to restart the experiment """

        partial_results = {}
        for k, v in self.results_dict.items():
            partial_results[k] = v if not isinstance(v, torch.Tensor) else v.cpu()

        checkpoint = {
            "model_weights": self.net.state_dict(),
            "torch_rng_state": torch.random.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "permutation_number": self.current_permutation,
            "current_running_avg_step": self.current_running_avg_step,
            "current_running_averages": (self.running_accuracy, self.running_loss),
            "partial_results": partial_results
        }

        if self.sparse_network:
            checkpoint["weight_masks"] = [m["mask"] for m in self.masks]

        if self.device.type == "cuda":
            checkpoint["cuda_rng_state"] = torch.cuda.get_rng_state()

        return checkpoint

    def load_checkpoint_data_and_update_experiment_variables(self, file_path):
        """ Loads the checkpoint and assigns the experiment variables the recovered values """

        with open(file_path, mode="rb") as experiment_checkpoint_file:
            checkpoint = pickle.load(experiment_checkpoint_file)

        self.net.load_state_dict(checkpoint["model_weights"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        np.random.set_state(checkpoint["numpy_rng_state"])
        self.current_permutation = checkpoint["permutation_number"]
        self.current_running_avg_step = checkpoint["current_running_avg_step"]
        self.running_accuracy, self.running_loss = checkpoint["current_running_averages"]

        if self.device.type == "cuda":
            torch.cuda.set_rng_state(checkpoint["cuda_rng_state"])

        partial_results = checkpoint["partial_results"]

        # store partial results
        for k, v in self.results_dict.items():
            self.results_dict[k] = partial_results[k] if not isinstance(partial_results[k], torch.Tensor) else \
            partial_results[k].to(self.device)

        if not self.sparse_network:
            return

        for i, mask in enumerate(self.masks):
            mask[i]["mask"] = checkpoint["weight_masks"][i].to(self.device)

    # ----------------------------- For storing summaries ----------------------------- #
    def _store_training_summaries(self):

        current_results = {
            "train_loss_per_checkpoint": self.running_loss / self.running_avg_window,
            "train_accuracy_per_checkpoint": self.running_accuracy / self.running_avg_window
        }

        # store train data for checkpoints
        self.results_dict["train_loss_per_checkpoint"][self.current_running_avg_step] += current_results["train_loss_per_checkpoint"]
        self.results_dict["train_accuracy_per_checkpoint"][self.current_running_avg_step] += current_results["train_accuracy_per_checkpoint"]

        self._print("\t\tOnline accuracy: {0:.2f}".format(self.running_accuracy / self.running_avg_window))
        self.running_loss *= 0.0
        self.running_accuracy *= 0.0
        self.current_running_avg_step += 1

    # --------------------------- For running the experiment --------------------------- #
    def run(self):
        # load data
        mnist_train_data = MnistDataSet(root_dir=self.data_path, train=True, device=self.device,
                                        image_normalization="max", label_preprocessing="one-hot", use_torch=True)
        mnist_data_loader = DataLoader(mnist_train_data, batch_size=self.batch_size, shuffle=True)

        # train network
        self.train(mnist_data_loader=mnist_data_loader, training_data=mnist_train_data)

    def train(self, mnist_data_loader: DataLoader, training_data: MnistDataSet):

        while self.current_permutation < self.num_permutations:
            initial_time = time.perf_counter()
            self._save_model_parameters()

            training_data.set_transformation(Permute(np.random.permutation(self.num_inputs)))  # apply new permutation
            print("\tPermutation number: {0}".format(self.current_permutation + 1))
            self.current_task_steps = 0
            for i, sample in enumerate(mnist_data_loader):
                if self.current_task_steps >= self.steps_per_task: break
                self.current_task_steps += 1
                self.current_experiment_step += 1

                # sample observation and target
                image = sample["image"].reshape(self.batch_size, self.num_inputs)
                label = sample["label"]

                # reset gradients
                for param in self.net.parameters(): param.grad = None  # apparently faster than optim.zero_grad()

                # compute prediction and loss
                predictions = self.net.forward(image)
                current_reg_loss = self.loss(predictions, label)
                current_loss = current_reg_loss.detach().clone()

                # backpropagate and update weights
                current_reg_loss.backward()
                self.optim.step()

                # update topology and apply masks to weights
                if self.time_to_update_topology(self.current_experiment_step):
                    self.update_topology()
                if self.sparsity > 0.0:
                    apply_weight_masks(self.masks)

                # store summaries
                current_accuracy = torch.mean((predictions.argmax(axis=1) == label.argmax(axis=1)).to(torch.float32))
                self.running_loss += current_loss
                self.running_accuracy += current_accuracy.detach()
                if (i + 1) % self.running_avg_window == 0:
                    self._print("\t\tStep Number: {0}".format(i + 1))
                    self._store_training_summaries()

            self.current_permutation += 1
            if self.current_permutation % self.checkpoint_save_frequency == 0:    # checkpoint experiment
                self.save_experiment_checkpoint()

            final_time = time.perf_counter()
            print("Epoch run time: {0:.2f}".format((final_time - initial_time) / 60))

        self._save_model_parameters()

    def time_to_update_topology(self, current_minibatch: int):
        if not self.use_dst:
            return False
        return (current_minibatch % self.topology_update_freq) == 0

    def update_topology(self):

        if self.current_task_steps >= self.max_steps_for_topology_update:
            self.current_topology_update += 1
            return
        removed_masks = []
        added_masks = []
        for mask in self.masks:
            if self.use_set_ds:
                third_arg = (mask["init_func"], self.dst_scale)
            else:
                if "num_active" not in mask.keys():
                    mask["num_active"] = mask["mask"].sum()
                third_arg = (int(self.drop_fraction * mask["num_active"]),)

            old_mask = deepcopy(mask["mask"])
            new_mask = self.dst_update_function(mask["mask"], mask["weight"], *third_arg)
            mask_difference = old_mask - new_mask
            added_masks.append(torch.clip(mask_difference, min=-1.0, max=0.0).abs())
            removed_masks.append(torch.clip(mask_difference, min=0.0, max=1.0))
            mask["mask"] = new_mask

        self.store_mask_update_summary(removed_masks, added_masks)
        self.current_topology_update += 1

    def store_mask_update_summary(self, removed_masks: list, added_masks: list):
        """
        Computes and stores the proportion of weights that were removed in the current topology update that were just
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
            # print("Total removed: {0}".format(total_removed))
            # print("Proportion of added then removed: {0:.4f}".format(prop_added_then_removed))
            self.results_dict["prop_added_then_removed"][self.current_topology_update] += prop_added_then_removed
            if self.use_set_ds:
                self.results_dict["total_removed_per_update"][self.current_topology_update] += total_removed

        self.previously_added_masks = added_masks

    def _save_model_parameters(self):
        """ Stores the parameters of the network """
        if not (self.current_permutation % self.parameter_save_frequency == 0) or not self.store_parameters:
            return
        model_parameters_dir_path = os.path.join(self.results_dir, "model_parameters")
        os.makedirs(model_parameters_dir_path, exist_ok=True)

        file_name = "index-{0}_task-{1}.pt".format(self.run_index, self.current_permutation)
        file_path = os.path.join(model_parameters_dir_path, file_name)

        masks = None if not self.sparse_network else [m["mask"] for m in self.masks]
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
    exp = PermutedMNISTExperiment(experiment_parameters,
                                  results_dir=os.path.join(file_path, "results", results_dir_name),
                                  run_index=terminal_arguments.run_index,
                                  verbose=terminal_arguments.verbose)
    exp.run()
    final_time = time.perf_counter()
    print("The running time in minutes is: {0:.2f}".format((final_time - initial_time) / 60))


if __name__ == "__main__":
    main()