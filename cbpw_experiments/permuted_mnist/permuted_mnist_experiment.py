# built-in libraries
import time
import os
import pickle

# third party libraries
import torch
from torch.utils.data import DataLoader
import numpy as np

# from ml project manager
from mlproj_manager.experiments import Experiment
from mlproj_manager.problems import MnistDataSet
from mlproj_manager.util import access_dict, Permute, get_random_seeds, turn_off_debugging_processes
from mlproj_manager.util.neural_networks import init_weights_kaiming
from mlproj_manager.file_management import store_object_with_several_attempts

# from src
from src.cbpw_functions import initialize_weight_dict
from src.networks import RegularizedSGD, ThreeHiddenLayerNetwork
from src.cbpw_functions.weight_matrix_updates import update_weights
from src.utils.experiment_utils import parse_terminal_arguments
from src.plasticity_functions import FirstOrderGlobalUPGD, inject_noise
from src.utils.evaluation_functions import compute_average_gradient_magnitude
from src.utils.permuted_mnist_experiment_utils import compute_average_weight_magnitude, compute_dead_units_proportion


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
        self.extended_summaries = access_dict(exp_params, "extended_summaries", default=False, val_type=bool)
        # learning parameters
        self.stepsize = exp_params["stepsize"]
        self.l1_factor = access_dict(exp_params, "l1_factor", default=0.0, val_type=float)
        self.l2_factor = access_dict(exp_params, "l2_factor", default=0.0, val_type=float)

        # architecture parameters
        self.num_hidden = exp_params["num_hidden"]      # number of hidden units per hidden layer
        self.batch_size = access_dict(exp_params, "batch_size", default=1, val_type=int)

        # problem parameters
        self.num_permutations = exp_params["num_permutations"]      # number of permutations (1 permutation = 1 epoch)
        self.steps_per_task = access_dict(exp_params, "steps_per_task", default=60000, val_type=int)
        assert self.steps_per_task <= 60000
        self.current_task_steps = 0
        self.current_experiment_step = 0

        # CBPw parameters
        self.use_cbpw = access_dict(exp_params, "use_cbpw", default=False, val_type=bool)
        self.topology_update_freq = access_dict(exp_params, "topology_update_freq", default=0, val_type=int)
        self.epoch_freq = access_dict(exp_params, "epoch_freq", default=False, val_type=bool)
        self.prune_method = access_dict(exp_params, "prune_method", default="none", val_type=str,
                                        choices=["none", "magnitude", "gf"])
        self.grow_method = access_dict(exp_params, "grow_method", default="none", val_type=str,
                                       choices=["none", "kaiming_normal", "zero", "fixed_with_noise"])
        self.drop_factor = access_dict(exp_params, "drop_factor", default=float, val_type=float)
        self.previously_removed_weights = None
        self.current_topology_update = 0

        # CBP parameters
        self.use_cbp = access_dict(exp_params, "use_cbp", default=False, val_type=bool)
        self.maturity_threshold = access_dict(exp_params, "maturity_threshold", default=0, val_type=int)
        self.replacement_rate = access_dict(exp_params, "replacement_rate", default=1e-6, val_type=float)

        # Layer Norm parameters
        self.use_ln = access_dict(exp_params, "use_ln", default=False, val_type=bool)
        self.preactivation_ln = access_dict(exp_params, "preactivation_ln", default=False, val_type=bool)

        # UPGD and S&P parameters
        self.use_upgd = access_dict(exp_params, "use_upgd", default=False, val_type=bool)
        self.perturb_weights = access_dict(exp_params, "perturb_weights", default=False, val_type=bool)
        self.noise_std = access_dict(exp_params, "noise_std", default=None, val_type=float)
        self.beta_utility = access_dict(exp_params, "beta_utility", default=0.0, val_type=float)

        # paths for loading and storing data
        self.data_path = exp_params["data_path"]
        self.store_parameters = access_dict(exp_params, "store_parameters", default=False, val_type=bool)
        self.parameter_save_frequency = 10  # how often to save the parameters in terms of number of tasks
        self.results_dir = results_dir

        """ Training constants """
        self.num_classes = 10
        self.num_inputs = 784
        self.max_num_images_per_permutation = 60000

        """ Network set up """
        # self.net = self.initialize_network()
        self.net = ThreeHiddenLayerNetwork(hidden_dim=self.num_hidden,
                                           use_cbp=self.use_cbp,
                                           maturity_threshold=self.maturity_threshold,
                                           replacement_rate=self.replacement_rate,
                                           use_layer_norm=self.use_ln,
                                           preactivation_layer_norm=self.preactivation_ln)
        self.net.apply(lambda z: init_weights_kaiming(z, nonlinearity="relu", normal=True))     # initialize weights

        # initialize CBPw dictionary
        self.weight_dict = None
        if self.use_cbpw:
            self.weight_dict = initialize_weight_dict(self.net, "sequential", self.prune_method,
                                                      self.grow_method, self.drop_factor, noise_std=self.noise_std)

        # initialize optimizer
        if self.use_upgd:
            self.optim = FirstOrderGlobalUPGD(self.net.named_parameters(),
                                              lr=self.stepsize,
                                              weight_decay=self.l2_factor / self.stepsize,
                                              beta_utility=self.beta_utility,
                                              sigma=self.noise_std)
        else:
            self.optim = RegularizedSGD(self.net.parameters(),
                                        lr=self.stepsize,
                                        weight_decay=self.l2_factor / self.stepsize,
                                        l1_reg_factor=self.l1_factor / self.stepsize)

        # define loss function
        self.loss = torch.nn.CrossEntropyLoss(reduction="mean")

        # move network to device
        self.net.to(self.device)

        """ Experiment Summaries """
        self.running_avg_window = 100 if self.batch_size == 1 else 10
        self.store_next_loss = False        # indicates whether to store the loss computed on the next batch
        self.current_running_avg_step, self.running_loss, self.running_accuracy, self.current_permutation = (0, 0.0, 0.0, 0)
        self.running_avg_grad_magnitude = 0.0
        self.results_dict = {}
        total_ckpts = self.steps_per_task * self.num_permutations // (self.running_avg_window * self.batch_size)
        self.results_dict["train_loss_per_checkpoint"] = torch.zeros(total_ckpts, device=self.device, dtype=torch.float32)
        self.results_dict["train_accuracy_per_checkpoint"] = torch.zeros(total_ckpts, device=self.device, dtype=torch.float32)

        if self.use_cbpw:
            total_top_updates = ((self.steps_per_task // self.batch_size) * self.num_permutations) // self.topology_update_freq
            self.results_dict["prop_added_then_removed"] = torch.zeros(total_top_updates, device=self.device, dtype=torch.float32)

        if self.extended_summaries:
            self.results_dict["average_gradient_magnitude_per_checkpoint"] = torch.zeros(total_ckpts, device=self.device, dtype=torch.float32)
            self.results_dict["average_weight_magnitude_per_permutation"] = torch.zeros(self.num_permutations, device=self.device, dtype=torch.float32)
            self.results_dict["proportion_dead_units_per_permutation"] = torch.zeros(self.num_permutations, device=self.device, dtype=torch.float32)

        if (self.use_cbp or self.use_cbpw) and self.extended_summaries:
            self.results_dict["loss_before_topology_update"] = []
            self.results_dict["loss_after_topology_update"] = []
            self.results_dict["avg_grad_before_topology_update"] = []
            self.results_dict["avg_grad_after_topology_update"] = []

        """ For creating experiment checkpoints """
        self.current_permutation = 0
        # with batch size of 30 and num permutations of 1000, experiment take less than an hour, so why checkpoints?
        self.store_checkpoints = False

    # ----------------------------- For storing summaries ----------------------------- #
    def _store_training_summaries(self):
        # store train data for checkpoints
        self.results_dict["train_loss_per_checkpoint"][self.current_running_avg_step] += self.running_loss / self.running_avg_window
        self.results_dict["train_accuracy_per_checkpoint"][self.current_running_avg_step] += self.running_accuracy / self.running_avg_window

        self._print("\t\tOnline accuracy: {0:.2f}".format(self.running_accuracy / self.running_avg_window))
        self.running_loss *= 0.0
        self.running_accuracy *= 0.0

        if self.extended_summaries:
            self.results_dict["average_gradient_magnitude_per_checkpoint"][self.current_running_avg_step] += \
                self.running_avg_grad_magnitude / self.running_avg_window
            self.running_avg_grad_magnitude *= 0.0

        self.current_running_avg_step += 1

    # --------------------------- For running the experiment --------------------------- #
    def run(self):
        # load data
        mnist_train_data = MnistDataSet(root_dir=self.data_path, train=True, device=self.device,
                                        image_normalization="max", label_preprocessing="one-hot", use_torch=True)
        mnist_data_loader = DataLoader(mnist_train_data, batch_size=self.batch_size, shuffle=True)

        # train network
        self.train(mnist_data_loader=mnist_data_loader, training_data=mnist_train_data)
        self.post_process_extended_results()

    def train(self, mnist_data_loader: DataLoader, training_data: MnistDataSet):

        while self.current_permutation < self.num_permutations:
            initial_time = time.perf_counter()
            self._save_model_parameters()

            training_data.set_transformation(Permute(np.random.permutation(self.num_inputs)))  # apply new permutation

            self.compute_network_extended_summaries(mnist_data_loader)

            print("\tPermutation number: {0}".format(self.current_permutation + 1))
            self.current_task_steps = 0
            for i, sample in enumerate(mnist_data_loader):
                if self.current_task_steps >= (self.steps_per_task // self.batch_size): break
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

                if self.extended_summaries:
                    self.running_avg_grad_magnitude += compute_average_gradient_magnitude(self.net)

                if self.perturb_weights:
                    inject_noise(self.net, noise_std=self.noise_std)

                self.store_extended_summaries(current_loss)

                # update topology and apply masks to weights
                if self.time_to_update_topology(self.current_experiment_step):
                    self.update_topology()

                # store summaries
                current_accuracy = torch.mean((predictions.argmax(axis=1) == label.argmax(axis=1)).to(torch.float32))
                self.running_loss += current_loss
                self.running_accuracy += current_accuracy.detach()
                if (i + 1) % self.running_avg_window == 0:
                    self._print("\t\tStep Number: {0}".format(i + 1))
                    self._store_training_summaries()

            self.current_permutation += 1
            if (self.current_permutation % self.checkpoint_save_frequency == 0) and self.store_checkpoints:
                self.save_experiment_checkpoint()

            final_time = time.perf_counter()
            print("Epoch run time: {0:.2f}".format((final_time - initial_time) / 60))

        self._save_model_parameters()

    def compute_network_extended_summaries(self, training_data: DataLoader):
        """ Computes the average weight magnitude and average dead units per permutation """

        if not self.extended_summaries: return
        avg_weight_magnitude = compute_average_weight_magnitude(self.net)
        prop_dead_units = compute_dead_units_proportion(self.net, training_data, self.num_hidden, self.batch_size)
        print(prop_dead_units)
        self.results_dict["average_weight_magnitude_per_permutation"][self.current_permutation] += avg_weight_magnitude
        self.results_dict["proportion_dead_units_per_permutation"][self.current_permutation] += prop_dead_units

    def store_extended_summaries(self, current_loss: torch.Tensor) -> None:
        """ Stores the extended summaries related to the topology update of CBP and CBPw """
        if not self.extended_summaries: return

        if (not self.store_cbp_extended_summaries() and         # check if using cbp and a feature has been replaced
            not self.store_cbpw_extended_summaries() and        # check if using cbpw and weights have been replaced
            not self.store_next_loss):                          # check if cbp or cbpw was used in the previous step
            return

        self.net.reset_indicators()
        prefix = "after" if self.store_next_loss else "before"
        self.results_dict[f"loss_{prefix}_topology_update"].append(current_loss)
        self.results_dict[f"avg_grad_{prefix}_topology_update"].append(self.net.get_average_gradient_magnitude())
        self.store_next_loss = not self.store_next_loss

    def store_cbp_extended_summaries(self) -> bool:
        return (self.use_cbp and self.net.feature_replace_event_indicator())

    def store_cbpw_extended_summaries(self) -> bool:
        return self.use_cbpw and self.time_to_update_topology(self.current_experiment_step)

    def time_to_update_topology(self, current_minibatch: int):
        if not self.use_cbpw:
            return False
        return (current_minibatch % self.topology_update_freq) == 0

    def update_topology(self):

        """
        Updates the neural network topology according to the chosen cbpw parameters
        """
        # update topology
        temp_summaries_dict = update_weights(self.weight_dict)
        # compute and store summaries
        removed_masks = [v[0] for v in temp_summaries_dict.values()]
        num_pruned = sum([v[1] for v in temp_summaries_dict.values()])
        self.store_mask_update_summary(removed_masks, num_pruned)

        self.current_topology_update += 1

    def store_mask_update_summary(self, removed_masks: list, total_removed: int):
        """
        Computes and stores the proportion of weights that were removed in the current topology update that were also
        removed in the last one

        Args:
            removed_masks: list of masks for weights that were removed in each layer for the current topology update
            total_removed: int corresponding to the total number of weights removed in the current topology update
        """

        if self.previously_removed_weights is not None:
            total_added_then_removed = 0
            for prev_removed, recently_removed in zip(self.previously_removed_weights, removed_masks):
                total_added_then_removed += ((prev_removed + recently_removed) == 0.0).sum()

            if total_removed == 0:
                prop_added_then_removed = 0.0
            else:
                prop_added_then_removed = total_added_then_removed / total_removed
            self.results_dict["prop_added_then_removed"][self.current_topology_update] += prop_added_then_removed

        self.previously_removed_weights = removed_masks

    def _save_model_parameters(self):
        """ Stores the parameters of the network """
        if not (self.current_permutation % self.parameter_save_frequency == 0) or not self.store_parameters:
            return
        model_parameters_dir_path = os.path.join(self.results_dir, "model_parameters")
        os.makedirs(model_parameters_dir_path, exist_ok=True)

        file_name = f"index-{self.run_index}.pt"
        file_path = os.path.join(model_parameters_dir_path, file_name)

        model_parameters = []
        if os.path.exists(file_path):
            if self.current_permutation == 0:   # there was something stored from previous failed runs
                os.remove(file_path)
            else:                               # there was something stored from the current run
                with open(file_path, mode="rb") as model_parameters_file:
                    model_parameters = pickle.load(model_parameters_file)
                os.remove(file_path)

        model_parameters.append(self.net.state_dict())
        store_object_with_several_attempts(model_parameters, file_path, storing_format="pickle", num_attempts=10)

    def post_process_extended_results(self):
        using_cbp_or_cbpw = self.use_cbp or self.use_cbpw
        if not self.extended_summaries or not using_cbp_or_cbpw: return
        self.results_dict["loss_before_topology_update"] = np.array(self.results_dict["loss_before_topology_update"], dtype=np.float32)
        self.results_dict["loss_after_topology_update"] = np.array(self.results_dict["loss_after_topology_update"], dtype=np.float32)
        self.results_dict["avg_grad_before_topology_update"] = np.array(self.results_dict["avg_grad_before_topology_update"], dtype=np.float32)
        self.results_dict["avg_grad_after_topology_update"] = np.array(self.results_dict["avg_grad_after_topology_update"], dtype=np.float32)


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
    exp.store_results()
    final_time = time.perf_counter()
    print("The running time in minutes is: {0:.2f}".format((final_time - initial_time) / 60))


if __name__ == "__main__":
    main()
