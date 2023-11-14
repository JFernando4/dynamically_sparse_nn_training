# built-in libraries
import time
import os

# third party libraries
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.prune import l1_unstructured, custom_from_mask, random_unstructured
import numpy as np
from sparselinear import SparseLinear

# from ml project manager
from mlproj_manager.problems import MnistDataSet, CifarDataSet
from mlproj_manager.experiments import Experiment
from mlproj_manager.function_approximators import GenericDeepNet
from mlproj_manager.util import turn_off_debugging_processes, get_random_seeds, access_dict, Permute
from mlproj_manager.util.neural_networks import get_optimizer, layer, xavier_init_weights


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


def get_sparse_mask_using_weight_magnitude(weights: torch.Tensor, sparsity_level: float):
    """
    Returns a mask of zeros and ones where only the weights above certain cutoff are assigned a value of one. The cutoff
    depends on the sparsity level.
    :param weights: tensor of weights
    :param sparsity_level: proportion of weights to prune out
    :return: tensor of same shape as weights
    """

    cutoff = torch.quantile(torch.abs(weights), sparsity_level)
    return (torch.abs(weights) > cutoff).to(torch.float32)


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
        mod.weights.multiply_(0.0)
        mod.weights.add_(weights[mask > 0.0])


class DynamicSparseMNISTExperiment(Experiment):

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
        self.num_epochs = access_dict(exp_params, "num_epochs", 1, val_type=int)
        self.num_layers = access_dict(exp_params, "num_layers", default=3, val_type=int)
        self.num_hidden = access_dict(exp_params, "num_hidden", 100, val_type=int)
        self.activation_function = access_dict(exp_params, "activation_function", "relu", val_type=str,
                                               choices=["relu", "leaky_relu", "sigmoid", "tanh"])
        self.permute_inputs = access_dict(exp_params, "permute_inputs", False, val_type=bool)
        self.plot = access_dict(exp_params, key="plot", default=False)

        assert 0.0 <= self.sparsity_level < 1.0
        self.sparsity_greater_than_zero = (self.sparsity_level > 0.0)
        assert 0.0 <= self.l1_factor
        self.use_l1_regularization = (self.l1_factor > 0.0)

        """ Training constants """
        self.batch_size = 1
        self.num_classes = 10
        self.num_inputs = (28, 28)
        self.max_num_steps = 60000

        """ Network set up """
        # initialize network architecture
        hidden_layer = layer(type="linear", parameters=(None, self.num_hidden), gate=self.activation_function)
        self.architecture = [hidden_layer for _ in range(self.num_layers)]
        self.architecture.append(layer(type="linear", parameters=(None, 10), gate=None))

        # initialize network
        self.net = torch.nn.Sequential()
        in_dims = np.prod(self.num_inputs)
        for l in range(self.num_layers):
            self.net.append(SparseLinear(in_dims, self.num_hidden, sparsity=self.sparsity_level))
            self.net.append(torch.nn.ReLU())
            in_dims = self.num_hidden
        self.net.append(torch.nn.Linear(self.num_hidden , 10))
        self.last_layer_index = len(self.net) - 1
        self._initialize_network()

        # initialize optimizer
        self.optim = get_optimizer("sgd", self.net.parameters(), stepsize=self.stepsize)

        # define loss function
        self.loss = torch.nn.CrossEntropyLoss(reduction="mean")

        # move network to device
        self.net.to(self.device)

        """ For summaries """
        self.checkpoint = 100
        self.current_ckpt, self.running_loss, self.running_accuracy, self.current_topology_update = (0, 0, 0, 0)
        self._initialize_summaries()

        # num_parameters_total = 0
        # num_parameters_pruned = 0
        # for i, mod in enumerate(self.net.network_module_list):
        #     if not hasattr(mod, "weight"): continue
        #
        #     num_parameters_total += mod.weight.numel() + mod.bias.numel()
        #     if i != self.last_layer_index:
        #         num_parameters_pruned += mod.weight_mask.sum() + mod.bias.numel()
        #     else:
        #         num_parameters_pruned += mod.weight.numel() + mod.bias.numel()
        #
        # print("Parameters in dense network: {0}\nParemeters in pruned network: {1}".format(num_parameters_total, num_parameters_pruned))

    # -------------------- Methods for initializing the experiment --------------------#
    def _initialize_network(self):
        """
        Initializes the network using xavier initialization
        """
        if self.sparsity_level == 0.0: return

        for i, mod in enumerate(self.net):

            # if last layer, initialize weights the usual way
            is_last_layer = (i == self.last_layer_index)
            if is_last_layer:
                xavier_init_weights(mod)
                continue

            # skip non-parametric layers
            is_non_parametric = not hasattr(mod, "weight")
            if is_non_parametric: continue

            new_weights = torch.zeros(mod.weight.size(), dtype=torch.float32)
            torch.nn.init.xavier_normal_(new_weights)

            new_mask = get_sparse_mask_using_weight_magnitude(new_weights, self.sparsity_level)
            copy_bias_and_weights_to_sparse_module(mod, torch.tensor(0.0), new_weights, new_mask)


    def _initialize_summaries(self):
        """
        Initializes the summaries for the experiment
        """
        total_checkpoints = self.max_num_steps * self.num_epochs // self.checkpoint
        self.results_dict["train_loss_per_checkpoint"] = torch.zeros(total_checkpoints, device=self.device,
                                                                     dtype=torch.float32)
        self.results_dict["train_accuracy_per_checkpoint"] = torch.zeros(total_checkpoints, device=self.device,
                                                                         dtype=torch.float32)
        if not self.permute_inputs:
            self.results_dict["test_loss_per_checkpoint"] = torch.zeros(self.num_epochs, device=self.device,
                                                                        dtype=torch.float32)
            self.results_dict["test_accuracy_per_checkpoint"] = torch.zeros(self.num_epochs, device=self.device,
                                                                            dtype=torch.float32)

        if self.sparsity_greater_than_zero:
            total_topology_updates = self.max_num_steps * self.num_epochs // self.topology_update_frequency
            num_hidden_layers = len(self.architecture) - 1      # the output layer is never pruned
            self.results_dict["num_units_pruned"] = torch.zeros((num_hidden_layers, total_topology_updates),
                                                                dtype=torch.float32)

    # ----------------------------- For storing summaries ----------------------------- #
    def _store_training_summaries(self):

        # store train data
        self.results_dict["train_loss_per_checkpoint"][self.current_ckpt] += self.running_loss / self.checkpoint
        self.results_dict["train_accuracy_per_checkpoint"][self.current_ckpt] += self.running_accuracy / self.checkpoint

        self._print("\t\tOnline accuracy: {0:.2f}".format(self.running_accuracy / self.checkpoint))
        self.running_loss *= 0.0
        self.running_accuracy *= 0.0
        self.current_ckpt += 1

    def _store_test_summaries(self, test_data, epoch_number: int):
        # store test data
        if not self.permute_inputs:
            test_loss, test_accuracy = self.evaluate_network(test_data)
            self.results_dict["test_loss_per_checkpoint"][epoch_number] += test_loss
            self.results_dict["test_accuracy_per_checkpoint"][epoch_number] += test_accuracy

    def evaluate_network(self, test_data: MnistDataSet):
        """
        Evaluates the network on the test data
        :param test_data: a pytorch DataSet object
        :return: (torch.Tensor) test loss, (torch.Tensor) test accuracy
        """
        with torch.no_grad():
            test_outputs = self.net.forward(test_data[:]["image"].reshape(-1, np.prod(self.num_inputs)))
            test_labels = test_data[:]["label"]

            loss = self.loss(test_outputs, test_labels)
            acc = torch.mean((test_outputs.argmax(axis=1) == test_labels.argmax(axis=1)).to(torch.float32))

        self._print("\t\tTest accuracy: {0:.4f}".format(acc))
        return loss, acc

    def _store_pruning_summaries(self, num_different: list):
        """
        stores the summaries about each topology update
        :param num_different: (list) number of different connections in each layer of the new network
        :param proportion_different: (list) proportion of different connections in each layer of the new network
        """
        for i in range(len(num_different)):
            self.results_dict["num_units_pruned"][i, self.current_topology_update] += num_different[i]

    # --------------------------- For running the experiment --------------------------- #
    def run(self):
        # load data
        training_data, training_dataloader = self.get_data(train=True, return_data_loader=True)
        test_data = self.get_data(train=False, return_data_loader=False)

        # train network
        self.train(mnist_data_loader=training_dataloader, test_data=test_data, training_data=training_data)

        self._plot_results()
        # summaries stored in memory automatically if using mlproj_manager

    def get_data(self, train: bool = True, return_data_loader: bool = False):
        """
        Loads the data set
        :param train: (bool) indicates whether to load the train (True) or the test (False) data
        :param return_data_loader: (bool) indicates whether to return a data loader object corresponding to the data set
        :return: data set, (optionally) data loader
        """
        """ Loads MNIST data set """
        mnist_data = MnistDataSet(root_dir=self.data_path,
                                  train=train,
                                  device=self.device,
                                  image_normalization="max",
                                  label_preprocessing="one-hot",
                                  use_torch=True)
        if return_data_loader:
            dataloader = DataLoader(mnist_data, batch_size=self.batch_size, shuffle=True)
            return mnist_data, dataloader

        return mnist_data

    def train(self, mnist_data_loader: DataLoader, test_data: MnistDataSet, training_data: MnistDataSet):

        for e in range(self.num_epochs):
            self._print("\tEpoch number: {0}".format(e + 1))

            for i, sample in enumerate(mnist_data_loader):
                # sample observationa and target
                image = sample["image"].reshape(self.batch_size, np.prod(self.num_inputs))
                label = sample["label"]

                # reset gradients
                for param in self.net.parameters(): param.grad = None  # apparently faster than optim.zero_grad()

                # compute prediction and loss
                predictions = self.net.forward(image)
                current_reg_loss = self.loss(predictions, label)
                current_loss = current_reg_loss.detach().clone()
                if self.use_l1_regularization:
                    current_reg_loss += self.l1_factor * torch.sum(torch.hstack([torch.abs(p).sum() for p in self.net.parameters()]))

                # backpropagate and update weights
                current_reg_loss.backward()
                self.optim.step()

                # store summaries
                current_accuracy = torch.mean((predictions.argmax(axis=1) == label.argmax(axis=1)).to(torch.float32))
                self.running_loss += current_loss
                self.running_accuracy += current_accuracy.detach()
                if (i + 1) % self.checkpoint == 0:
                    self._print("\t\tStep Number: {0}".format(i + 1))
                    self._store_training_summaries()

                # update topology
                self._inject_noise_and_prune(step=i + 1)

            self._store_test_summaries(test_data, epoch_number=e)

            if self.permute_inputs:
                training_data.set_transformation(Permute(np.random.permutation(np.arange(np.prod(self.num_inputs)))))

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
            raise NotImplemented #TODO
            new_net, num_different_per_layer = self._inject_noise_and_prune_global()
        else:
            new_net, num_different_per_layer = self._inject_noise_and_prune_local()

        self._update_current_network_variables(new_net)
        self._store_pruning_summaries(num_different_per_layer)
        self.current_topology_update += 1

    def _inject_noise_and_prune_local(self):
        """
        Updates the topology of the network by replacing weights whose magnitude is lower than the magnitude of random
        weights sampled from the initial distribution. Specifically, the topology update follows these steps:
            1. create a new network where:
                hidden layers: weights are randomly initialized and bias are copied from the current learning net
                output layer: weights and bias are copied from the current learning net
            2. the weights of the current pruned learning network are inserted into the new network
            3. each hidden layer of the new network is pruned using weight magnitude pruning
            4. the current network is updated to be the new network and a new optimizer is initialized
        :return: None, but stores the number of weights that are different from the previous network
        """

        # initialize new network
        new_net = torch.nn.Sequential()

        num_different_per_layer = []
        for i, mod in enumerate(self.net):

            # append current last layer to the new net
            is_last_layer = (i == self.last_layer_index)
            if is_last_layer:
                new_net.append(mod)
                continue

            # skip layers that don't have any parameters
            is_non_parametric = not hasattr(mod, "weight")
            if is_non_parametric:
                new_net.append(mod)
                continue

            current_mask = get_mask_from_sparse_module(mod)
            # sanity check:
            # torch.all(current_mask.to_sparse().indices() == mod.weight.indices())

            new_weights = get_dense_weights_from_sparse_module(mod, current_mask)
            # sanity check
            # torch.all(new_weights[mod.weight.indices()[0], mod.weight.indices()[1]] == mod.weight.values())

            new_mask = get_sparse_mask_using_weight_magnitude(new_weights, self.sparsity_level)
            new_connections = new_mask.to_sparse().indices()

            new_sparse_mod = SparseLinear(mod.in_features, mod.out_features, connectivity=new_connections)
            copy_bias_and_weights_to_sparse_module(new_sparse_mod, mod.bias, new_weights, new_mask)

            # sanity checks:
            # print(torch.all(new_sparse_mod.bias == mod.bias))    # should be True
            # print(torch.all(new_sparse_mod.weight.values() == new_weights[new_mask > 0.0]))  # should return True
            new_net.append(new_sparse_mod)

            # store relevant summaries
            num_different = torch.sum(torch.clip(current_mask - new_mask, 0.0, 1.0))
            num_different_per_layer.append(num_different)
            # self._print("Number of different entries in mask: {0}".format(num_different))

        return new_net, num_different_per_layer

    def _inject_noise_and_prune_global(self):
        """
        Updates the topology of the network by replacing weights whose magnitude is lower than the magnitude of random
        weights sampled from the initial distribution. Specifically, the topology is updated using the following steps:
            1. Create a new network where:
                hidden layers: bias are copied from the current learning net, weights are initialized to zero
                output layer: bias and weight matrix are copied from the current learning net
            2. all the weights of the hidden layers of the current learning network are stacked, same for the masks
            3. a new weight tensor is randomly initialized with a normal(0, 1/sqrt(num_inputs + num_outputs))
            4. old weights are inserted back into the randomly initialized new weights
            5. the cutoff for the give sparsity level is calculated using all the weights in the new weight tensor, and
               a new mask is obtained
            6. the weights and masks are then copied into the new network
            7. the current network is updated to be the new network and a new optimizer is instantiated
        :return: None, but stores the number of weights that are different from the previous network
        """

        # initialize new network
        new_net = GenericDeepNet(self.net.architecture, self.num_inputs)

        weight_list = []
        mask_list = []

        for i, mod in enumerate(self.net.network_module_list):
            is_non_parametric = not hasattr(mod, "weight")
            if is_non_parametric: continue

            # copy bias to the new network
            with torch.no_grad():
                new_net.network_module_list[i].bias.multiply_(0.0)
                new_net.network_module_list[i].bias.add_(mod.bias)

            # if it's the last layer, also copy the weights to the new network
            is_last_layer = (i == self.last_layer_index)
            if is_last_layer:
                with torch.no_grad():
                    new_net.network_module_list[i].weight.multiply_(0.0)
                    new_net.network_module_list[i].weight.add_(mod.weight)
                continue

            # retrieve the weights and masks for any layer other than the output layer
            weight_list.append(mod.weight.flatten())
            mask_list.append(mod.weight_mask.flatten())

        flat_weight_list = torch.hstack(weight_list)
        flat_mask_list = torch.hstack(mask_list)

        # initialize new weights
        new_weights = torch.zeros_like(flat_weight_list)
        torch.nn.init.normal_(new_weights, mean=0.0, std=1/np.sqrt(np.prod(self.num_inputs) + self.num_classes))

        # insert old weights into the array of new weights
        negative_mask = 1.0 - flat_mask_list
        new_weights *= negative_mask
        new_weights += flat_weight_list

        # find the cutoff for the new weights and create new mask
        cut_off = torch.quantile(torch.abs(new_weights), self.sparsity_level)
        new_masks = (torch.abs(new_weights) >= cut_off).to(torch.float32)

        num_different_per_layer = []
        current_index = 0
        for i, mod in enumerate(new_net.network_module_list):
            is_non_parametric = not hasattr(mod, "weight")
            is_last_layer = (i == self.last_layer_index)
            if is_last_layer or is_non_parametric: continue

            # retrieve and reshape new weights and mask for the layer
            numel_in_layer = mod.weight.numel()
            weight_shape = mod.weight.shape
            new_layer_weights = new_weights[current_index: current_index + numel_in_layer].reshape(weight_shape)
            new_layer_mask = new_masks[current_index: current_index + numel_in_layer].reshape(weight_shape)

            # copy weights to new network
            with torch.no_grad():
                mod.weight.multiply_(0.0)
                mod.weight.add_(new_layer_weights)

            # prune network
            custom_from_mask(mod, "weight", mask=new_layer_mask)
            current_index += numel_in_layer

            # store relevant summaries
            old_layer_mask = self.net.network_module_list[i].weight_mask
            num_different = torch.sum(torch.clip(mod.weight_mask - old_layer_mask, 0.0, 1.0))
            num_different_per_layer.append(num_different)
            self._print("Number of different entries in mask: {0}".format(num_different))

        return new_net, num_different_per_layer

    def _update_current_network_variables(self, new_net: GenericDeepNet):
        """
        Updates the internal variables corresponding to the network and the optimizer
        :param new_net: an instance of GenericDeepNet
        """
        self.net = new_net
        self.optim = get_optimizer("sgd", self.net.parameters(), stepsize=self.stepsize)
        self.net.to(self.device)

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
                    plt.ylim((0,1))
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
        "stepsize": 0.01,   # 0.01 for mnist, 0.0005 for cifar 10
        "l1_factor": 0.0001,    # 0.0001 for mnist
        "topology_update_frequency": 200,
        "sparsity_level": 0.0,
        "global_pruning": False,
        "data_path": os.path.join(file_path, "data"),
        "num_epochs": 100,
        "num_layers": 3,
        "num_hidden": 100,
        "activation_function": "relu",
        "permute_inputs": True,
        "plot": True
    }

    print(experiment_parameters)
    initial_time = time.perf_counter()
    exp = DynamicSparseMNISTExperiment(experiment_parameters,
                                       results_dir=os.path.join(file_path, "results"),
                                       run_index=0,
                                       verbose=True)
    exp.run()
    exp.store_results()
    final_time = time.perf_counter()
    print("The running time in minutes is: {0:.2f}".format((final_time - initial_time) / 60))


if __name__ == "__main__":
    main()
