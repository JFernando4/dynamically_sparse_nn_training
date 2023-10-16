# built-in libraries
import time
import os

# third party libraries
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.prune import custom_from_mask
import numpy as np
from scipy.stats import norm, truncnorm
from sparselinear import SparseLinear

# from ml project manager
from mlproj_manager.problems.reinforcement_learning import MountainCar
from mlproj_manager.experiments import Experiment
from mlproj_manager.util import turn_off_debugging_processes, get_random_seeds, access_dict, Permute
from mlproj_manager.util.neural_networks import get_optimizer, xavier_init_weights, get_activation_module
from mlproj_manager.util.neural_networks.weights_initialization_and_manipulation import apply_regularization_to_module

from src.utils import get_mask_from_sparse_module, get_dense_weights_from_sparse_module, \
    get_sparse_mask_using_weight_magnitude, copy_bias_and_weights_to_sparse_module, copy_weights_to_sparse_module


class DynamicSparseMCExperiment(Experiment):

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
        np.random.seed(self.random_seed)

        """ Experiment parameters """
        self.stepsize = exp_params["stepsize"]
        self.l1_factor = exp_params["l1_factor"]
        self.topology_update_frequency = exp_params["topology_update_frequency"]
        self.sparsity_level = exp_params["sparsity_level"]
        self.global_pruning = access_dict(exp_params, "global_pruning", default=False, val_type=bool)

        self.data_path = exp_params["data_path"]
        self.num_layers = access_dict(exp_params, "num_layers", default=3, val_type=int)
        self.num_hidden = access_dict(exp_params, "num_hidden", default=100, val_type=int)
        self.activation_function = access_dict(exp_params, "activation_function", default="relu", val_type=str,
                                               choices=["relu", "leaky_relu", "sigmoid", "tanh"])
        self.permute_inputs = access_dict(exp_params, "permute_inputs", False, val_type=bool)
        self.sparsify_last_layer = access_dict(exp_params, "sparsify_last_layer", default=False, val_type=bool)
        self.plot = access_dict(exp_params, key="plot", default=False)

        assert 0.0 <= self.sparsity_level < 1.0
        self.sparsity_greater_than_zero = (self.sparsity_level > 0.0)
        assert 0.0 <= self.l1_factor
        self.use_l1_regularization = (self.l1_factor > 0.0)

        """ Training constants """
        self.batch_size = 1
        self.gamma = 0.99
        self.epsilon = 0.1
        self.num_classes = 3
        self.num_inputs = 2
        self.max_num_steps = 200000

        """ Network set up """
        # initialize network
        self.net = self._initialize_network_architecture()
        self.last_layer_index = len(self.net) - 1
        self.traces = [torch.zeros_like(p) for p in self.net.parameters()]

        # initialize optimizer
        self.optim = get_optimizer("sgd", self.net.parameters(), stepsize=self.stepsize)

        # define loss function
        self.loss = torch.nn.MSELoss(reduction="mean")

        # move network to device
        self.net.to(self.device)

        """ For summaries """
        self.checkpoint = 1000
        self.current_ckpt, self.running_loss, self.running_sum_of_rewards, self.current_topology_update = (0, 0, 0, 0)
        self._initialize_summaries()

    # -------------------- Methods for initializing the experiment --------------------#
    def _initialize_network_architecture(self):
        """
        Initializes the torch module representing the neural network
        """

        if not self.sparsity_greater_than_zero:
            return self._initialize_dense_network_architecture()
        else:
            return self._initialize_sparse_network_architecture()

    def _initialize_dense_network_architecture(self):
        """
        Initializes the torch module representing a dense feedforward network
        """
        net = torch.nn.Sequential()
        in_dims = self.num_inputs

        # initialize hidden layers
        for _ in range(self.num_layers):
            # initialize module
            current_module = torch.nn.Linear(in_dims, self.num_hidden)
            # initialize module weights
            xavier_init_weights(current_module)
            # append module and activation function
            net.append(current_module)
            net.append(get_activation_module(self.activation_function))
            in_dims = self.num_hidden

        # initialize output layer, output layer weights, and append it to the network
        output_layer = torch.nn.Linear(self.num_hidden, self.num_classes)
        xavier_init_weights(output_layer)
        net.append(output_layer)

        return net

    def _initialize_sparse_network_architecture(self):
        """
        Initializes the torch module representing a sparse feedforward network
        """
        net = torch.nn.Sequential()
        in_dims = self.num_inputs

        # initialize hidden layers
        for _ in range(self.num_layers):
            # initialize module
            current_module = SparseLinear(in_dims, self.num_hidden, sparsity=self.sparsity_level)
            # initialize module weights
            new_weights = torch.zeros(current_module.weight.size(), dtype=torch.float32)
            torch.nn.init.xavier_normal_(new_weights)
            new_mask = get_sparse_mask_using_weight_magnitude(new_weights, k=current_module.weight.values().size()[0])
            copy_weights_to_sparse_module(current_module, new_weights[new_mask > 0.0])
            # copy_bias_and_weights_to_sparse_module(current_module, bias=torch.tensor(0.0), weights=new_weights, mask=new_mask)
            # append module and activation function
            net.append(current_module)
            net.append(get_activation_module(self.activation_function))
            in_dims = self.num_hidden

        # initialize output layer, output layer weights, and append it to the network
        if self.sparsify_last_layer:
            output_module = SparseLinear(self.num_hidden, self.num_classes, sparsity=self.sparsity_level)
            new_weights = torch.zeros(output_module.weight.size(), dtype=torch.float32)
            torch.nn.init.xavier_normal_(new_weights)
            new_mask = get_sparse_mask_using_weight_magnitude(new_weights, k=output_module.weight.values().size()[0])
            copy_bias_and_weights_to_sparse_module(output_module, bias=torch.tensor(0.0), weights=new_weights, mask=new_mask)
        else:
            output_module = torch.nn.Linear(self.num_hidden, self.num_classes)
            xavier_init_weights(output_module)
        net.append(output_module)

        return net

    def _initialize_summaries(self):
        """
        Initializes the summaries for the experiment
        """
        total_checkpoints = self.max_num_steps // self.checkpoint
        self.results_dict["train_loss_per_checkpoint"] = torch.zeros(total_checkpoints, device=self.device,
                                                                     dtype=torch.float32)
        self.results_dict["return_per_episode"] = []

        if self.sparsity_greater_than_zero:
            total_topology_updates = self.max_num_steps // self.topology_update_frequency
            num_layers = self.num_layers if not self.sparsify_last_layer else self.num_layers + 1
            self.results_dict["num_weights_pruned"] = torch.zeros((num_layers, total_topology_updates),
                                                                  dtype=torch.float32)

    # ----------------------------- For storing summaries ----------------------------- #
    def _store_training_summaries(self):

        # store train data
        self.results_dict["train_loss_per_checkpoint"][self.current_ckpt] += self.running_loss / self.checkpoint

        self._print("\t\tOnline Sum of Rewards: {0:.2f}".format(self.running_sum_of_rewards / self.checkpoint))
        self.running_loss *= 0.0
        self.running_sum_of_rewards *= 0.0
        self.current_ckpt += 1

    def evaluate_network(self):
        """
        Evaluates the network on the test data
        :param test_data: a pytorch DataSet object
        :return: (torch.Tensor) test loss, (torch.Tensor) test accuracy
        """
        return

    def _store_pruning_summaries(self, num_different: list):
        """
        stores the summaries about each topology update
        :param num_different: (list) number of different connections in each layer of the new network
        :param proportion_different: (list) proportion of different connections in each layer of the new network
        """
        for i in range(len(num_different)):
            self.results_dict["num_weights_pruned"][i, self.current_topology_update] += num_different[i]

    # --------------------------- For running the experiment --------------------------- #
    def run(self):
        # load data
        env = MountainCar(normalize_state=True)

        # traces = []
        # for p in self.net.parameters():
        #     traces.append(torch.zeros_like(p, requires_grad=False))
        lambda_factor = 0.95

        num_terminations = 0
        # train network
        env.reset()
        current_state = env.get_current_state()
        current_action_values = self.net.forward(torch.tensor(current_state, dtype=torch.float32))
        with torch.no_grad():
            p = np.random.rand()
            # epsilon greed policy
            if p < self.epsilon:
                current_action = torch.tensor(np.random.randint(0, 3))
            else:
                current_action = torch.argmax(current_action_values)

        for step in range(self.max_num_steps):

            # reset gradients
            for param in self.net.parameters(): param.grad = None  # apparently faster than optim.zero_grad()

            # sample state
            # current_state = env.get_current_state()

            # get action values and select new action
            current_action_values = self.net.forward(torch.tensor(current_state, dtype=torch.float32))
            if torch.any(torch.isnan(current_action_values)):
                print("The algorithm diverged!")
                break
            # with torch.no_grad():
            #     p = np.random.rand()
            #     epsilon greed policy
            #     if p < self.epsilon:
            #         current_action = torch.tensor(np.random.randint(0,3))
            #     else:
            #         current_action = torch.argmax(current_action_values)

            # execute action
            next_state, reward, termination_signal = env.step(int(current_action))

            # get next action and action values
            with torch.no_grad():
                next_action_values = self.net.forward(torch.tensor(next_state, dtype=torch.float32))
                p = np.random.rand()
                if p < self.epsilon:
                    next_action = torch.tensor(np.random.randint(0, 3))
                else:
                    next_action = torch.argmax(next_action_values)

            # compute td error
            continuation_prob = self.gamma * (1.0 - float(termination_signal))
            td_error = reward + continuation_prob * next_action_values.gather(0, next_action) - current_action_values.gather(0, current_action)

            td_error.backward()

            # update eligibility_traces
            for i, param in enumerate(self.net.parameters()):
                self.traces[i] *= self.gamma * lambda_factor
                self.traces[i] -= param.grad

            # update parameters
            with torch.no_grad():
                for i, param in enumerate(self.net.parameters()):
                    param.add_(self.stepsize * td_error * self.traces[i])

            reg_squared_td_error = torch.square(td_error)
            squared_td_error = reg_squared_td_error.detach().clone()
            # reg_squared_td_error.backward()
            # self.optim.step()

            # apply regularization
            if self.use_l1_regularization:
                for i, module in enumerate(self.net):
                    is_non_parametric = not hasattr(module, "weight")
                    if is_non_parametric: continue

                    if i != len(self.net) - 1:
                        if hasattr(module, "weights"):
                            apply_regularization(module, "weights", l1_factor=self.l1_factor)
                        else:
                            apply_regularization(module, "weight", l1_factor=self.l1_factor)

            self.running_sum_of_rewards += reward
            self.running_loss += squared_td_error

            if (step + 1) % self.checkpoint == 0:
                self._print("\t\tStep Number: {0}".format(step + 1))
                self._print("\t\tCurrent action values: {0}".format(current_action_values))
                self._store_training_summaries()
                print("\t\tNumber of terminations: {0}".format(len(self.results_dict["return_per_episode"])))

            # update topology
            self._inject_noise_and_prune(step=step + 1)
            # matches_frequency = ((step + 1 % self.topology_update_frequency) == 0)
            # time_to_update_topology = matches_frequency and self.sparsity_greater_than_zero
            # if time_to_update_topology:
            #     # todo: only resets the traces of the weights that got disconnected from the network
            #     for i, tr in enumerate(traces):
            #         if i < len(traces) - 2:
            #             tr.multiply_(0.0)

            current_state = next_state
            current_action = next_action

            if termination_signal:
                num_terminations += 1
                self.results_dict["return_per_episode"].append(self.running_sum_of_rewards)
                self.running_sum_of_rewards *= 0.0

                env.reset()
                current_state = env.get_current_state()
                current_action_values = self.net.forward(torch.tensor(current_state, dtype=torch.float32))
                with torch.no_grad():
                    p = np.random.rand()
                    # epsilon greed policy
                    if p < self.epsilon:
                        current_action = torch.tensor(np.random.randint(0, 3))
                    else:
                        current_action = torch.argmax(current_action_values)

                for tr in self.traces:
                    tr.multiply_(0.0)

        self.results_dict["return_per_episode"] = torch.tensor(self.results_dict["return_per_episode"])
        # self._plot_results()
        # summaries stored in memory automatically if using mlproj_manager

    def _inject_noise_and_prune(self, step: int):
        """
        Updates the topology of the network by doing local or global pruning
        :param step: the current training step
        :return: None, but stores summaries relevant to the pruning step
        """

        # check if topology should be updated
        matches_frequency = ((step % self.topology_update_frequency) == 0)
        time_to_update_topology = matches_frequency and self.sparsity_greater_than_zero
        if not time_to_update_topology:return

        # do global or local pruning and update topology with new random connections
        if self.global_pruning:
            raise NotImplemented
        else:
            # new_net, num_different_per_layer = self._inject_noise_and_prune_local()
            num_different_per_layer = self._inject_noise_and_prune_local_efficient()

        self._store_pruning_summaries(num_different_per_layer)
        self.current_topology_update += 1

    def _inject_noise_and_prune_local_efficient(self):
        """

        :return:
        """

        trace_index = 0
        num_different_per_layer = []
        for i, mod in enumerate(self.net):

            # if the last layer is not sparse, then continue
            is_last_layer = (i == self.last_layer_index)
            if is_last_layer and (not self.sparsify_last_layer): continue

            # skip layers that don't have any parameters
            is_non_parametric = not hasattr(mod, "weight")
            if is_non_parametric: continue

            new_traces = torch.zeros(mod.weight.size(), dtype=torch.float32, device=self.device)
            new_traces[mod.weight.indices()[0], mod.weight.indices()[1]] += self.traces[trace_index]

            current_mask = get_mask_from_sparse_module(mod)
            # sanity check:
            # torch.all(current_mask.to_sparse().indices() == mod.weight.indices())

            new_weights = get_dense_weights_from_sparse_module(mod, current_mask)
            # sanity check
            # torch.all(new_weights[mod.weight.indices()[0], mod.weight.indices()[1]] == mod.weight.values())

            new_mask = get_sparse_mask_using_weight_magnitude(new_weights, k=mod.weight.values().size()[0])
            # sanity check
            # print(current_mask.sum() == new_mask.sum())

            masked_weights = new_weights * new_mask
            sparse_weights = masked_weights.to_sparse()

            mod.indices = sparse_weights.indices()
            copy_weights_to_sparse_module(mod, sparse_weights.values())
            # sanity checks:
            # print(torch.all(mod.weight.indices() == sparse_weights.indices()))
            # print(torch.all(mod.weight.values() == sparse_weights.values()))

            # store relevant summaries
            num_different = torch.sum(torch.clip(current_mask - new_mask, 0.0, 1.0))
            num_different_per_layer.append(num_different)
            # self._print("Number of different entries in mask: {0}".format(num_different))

            # update traces
            self.traces[trace_index] = new_traces[mod.weight.indices()[0], mod.weight.indices()[1]]
            trace_index += 2

        return num_different_per_layer

    def _plot_results(self):
        if self.plot:
            import matplotlib.pyplot as plt

            for rname, rvals in self.results_dict.items():
                if "pruned" in rname:
                    for i in range(rvals.shape[0]):
                        plt.plot(torch.arange(rvals.shape[1]), rvals[i])
                else:
                    plt.plot(torch.arange(rvals.size()[0]), rvals)
                if "accuracy" in rname:
                    plt.ylim((0.8,1))
                plt.title(rname)
                plt.show()
                plt.close()


def main():
    """
    This is a quick demonstration of how to run the experiments. For a more systematic run, use the mlproj_manager
    scheduler.
    """
    file_path = os.path.dirname(os.path.abspath(__file__))
    experiment_parameters = {
        "stepsize": 0.0001,   # 0.0001 for DST alg
        "l1_factor": 1e-7,    # 0.1 for DST alg
        "topology_update_frequency": 200,
        "sparsity_level": 0.9,  # 0.9
        "global_pruning": False,
        "data_path": os.path.join(file_path, "data"),
        "num_epochs": 200,
        "num_layers": 2,
        "num_hidden": 100,
        "activation_function": "relu",
        "permute_inputs": False,
        "sparsify_last_layer": False,
        "plot": True
    }

    print(experiment_parameters)
    relevant_parameters = ["topology_update_frequency", "l1_factor", "sparsity_level", "num_epochs", "num_layers",
                           "num_hidden", "sparsify_last_layer"]
    results_dir_name = ""
    for relevant_param in relevant_parameters:
        results_dir_name += "_" + relevant_param + "-" + str(experiment_parameters[relevant_param])

    num_seeds = 30
    for i in range(num_seeds):
        initial_time = time.perf_counter()
        exp = DynamicSparseMCExperiment(experiment_parameters,
                                        results_dir=os.path.join(file_path, "results", results_dir_name),
                                        run_index=i,
                                        verbose=True)
        exp.run()
        exp.store_results()
        final_time = time.perf_counter()
        print("The running time in minutes is: {0:.2f}".format((final_time - initial_time) / 60))


if __name__ == "__main__":
    main()
