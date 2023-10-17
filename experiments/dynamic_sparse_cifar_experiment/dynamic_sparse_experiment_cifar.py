# built-in libraries
import time
import os

# third party libraries
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn.utils.prune import l1_unstructured, custom_from_mask, random_unstructured
import numpy as np
from scipy.stats import norm, truncnorm
from sparselinear import SparseLinear

# from ml project manager
from mlproj_manager.problems import CifarDataSet
from mlproj_manager.experiments import Experiment
from mlproj_manager.util import turn_off_debugging_processes, get_random_seeds, access_dict, Permute
from mlproj_manager.util.neural_networks import get_optimizer, xavier_init_weights, get_activation_module
from mlproj_manager.util.neural_networks.weights_initialization_and_manipulation import apply_regularization_to_module, apply_regularization_to_tensor
from mlproj_manager.util.data_preprocessing_and_transformations.image_transformations import GrayScale, \
    RandomHorizontalFlip, RandomRotator, RandomErasing

# source files
from src.utils import get_mask_from_sparse_module, get_dense_weights_from_sparse_module, \
    get_sparse_mask_using_weight_magnitude, copy_bias_and_weights_to_sparse_module, copy_weights_to_sparse_module


class DynamicSparseCIFARExperiment(Experiment):

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
        self.l2_factor = exp_params["l2_factor"]
        self.topology_update_frequency = exp_params["topology_update_frequency"]
        self.sparsity_level = exp_params["sparsity_level"]
        self.global_pruning = access_dict(exp_params, "global_pruning", default=False, val_type=bool)

        self.data_path = exp_params["data_path"]
        self.num_epochs = access_dict(exp_params, "num_epochs", default=1, val_type=int)
        self.num_layers = access_dict(exp_params, "num_layers", default=3, val_type=int)
        self.num_hidden = access_dict(exp_params, "num_hidden", default=100, val_type=int)
        self.activation_function = access_dict(exp_params, "activation_function", default="relu", val_type=str,
                                               choices=["relu", "leaky_relu", "sigmoid", "tanh"])
        self.permute_inputs = access_dict(exp_params, "permute_inputs", False, val_type=bool)
        self.sparsify_last_layer = access_dict(exp_params, "sparsify_last_layer", default=False, val_type=bool)
        self.reverse_transformation_order = access_dict(exp_params, "reverse_transformation_order",
                                                        default=False, val_type=bool)
        self.plot = access_dict(exp_params, key="plot", default=False)

        assert 0.0 <= self.sparsity_level < 1.0
        self.sparsity_greater_than_zero = (self.sparsity_level > 0.0)
        assert 0.0 <= self.l1_factor
        self.use_l1_regularization = (self.l1_factor > 0.0)
        assert 0.0 <= self.l2_factor
        self.use_l2_regularization = (self.l1_factor > 0.0)
        self.use_regularization = self.use_l1_regularization or self.use_l2_regularization

        """ Training constants """
        self.batch_size = 100
        self.num_classes = 10
        self.image_dims = (32, 32 * 3)
        self.flat_image_dims = np.prod(self.image_dims)
        self.num_images_per_epoch = 50000
        self.num_test_samples = 10000
        self.num_test_batches = self.num_test_samples / self.batch_size
        self.is_conv = False

        """ Network set up """
        # initialize network
        self.net = self._initialize_network_architecture()
        self.last_layer_index = len(self.net) - 1
        self.regularization_indicators = self._get_regularization_indicators()

        # initialize optimizer
        self.optim = torch.optim.SGD(self.net.parameters(), lr=self.stepsize)

        # define loss function
        self.loss = torch.nn.CrossEntropyLoss(reduction="mean")

        # move network to device
        self.net.to(self.device)

        """ For summaries """
        self.checkpoint = 100
        self.current_ckpt, self.running_loss, self.running_accuracy, self.running_regularized_loss, \
            self.current_topology_update = (0, 0, 0, 0, 0)
        self._initialize_summaries()

        """ For non-stationary data transformations """
        self.num_transformations = 0
        self.scale_increase = 0.03 #0.05
        self.rotation_increase = 14.4 # 7.2 #3.5
        self.transformations = self._get_transformations()

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
        in_dims = int(np.prod(self.image_dims))

        # initialize hidden layers
        num_hidden = [self.num_hidden] * self.num_layers
        num_hidden[self.num_layers // 2] = self.num_hidden // 4
        for i in range(self.num_layers):
            # initialize module
            current_module = torch.nn.Linear(in_dims, num_hidden[i])
            # initialize module weights
            xavier_init_weights(current_module)
            # append module and activation function
            net.append(current_module)
            net.append(get_activation_module(self.activation_function))
            net.append(torch.nn.Dropout(p=0.3))
            in_dims = num_hidden[i]

        # initialize output layer, output layer weights, and append it to the network
        output_layer = torch.nn.Linear(num_hidden[-1], self.num_classes)
        xavier_init_weights(output_layer)
        net.append(output_layer)

        return net

    def _initialize_sparse_network_architecture(self):
        """
        Initializes the torch module representing a sparse feedforward network
        """
        net = torch.nn.Sequential()
        in_dims = int(np.prod(self.image_dims))
        # num_hidden = [4000, 1000, 4000]
        # initialize hidden layers
        num_hidden = [self.num_hidden] * self.num_layers
        num_hidden[self.num_layers // 2] = self.num_hidden // 4
        for i in range(self.num_layers):
            # initialize module
            current_module = SparseLinear(in_dims, num_hidden[i], sparsity=self.sparsity_level)
            # initialize module weights
            new_weights = torch.zeros(current_module.weight.size(), dtype=torch.float32)
            torch.nn.init.xavier_normal_(new_weights)
            new_mask = get_sparse_mask_using_weight_magnitude(new_weights, k=current_module.weight.values().size()[0])
            copy_weights_to_sparse_module(current_module, new_weights[new_mask > 0.0])
            # append module and activation function
            net.append(current_module)
            net.append(get_activation_module(self.activation_function))
            in_dims = num_hidden[i]

        # initialize output layer, output layer weights, and append it to the network
        if self.sparsify_last_layer:
            output_module = SparseLinear(num_hidden[-1], self.num_classes, sparsity=self.sparsity_level)
            new_weights = torch.zeros(output_module.weight.size(), dtype=torch.float32)
            torch.nn.init.xavier_normal_(new_weights)
            new_mask = get_sparse_mask_using_weight_magnitude(new_weights, k=output_module.weight.values().size()[0])
            copy_bias_and_weights_to_sparse_module(output_module, bias=torch.tensor(0.0), weights=new_weights, mask=new_mask)
        else:
            output_module = torch.nn.Linear(num_hidden[-1], self.num_classes)
            xavier_init_weights(output_module)
        net.append(output_module)

        return net

    def _get_regularization_indicators(self):
        """
        Creates a list of tuples that indicate if the corresponding layer has regularization applied to it and the
        name of the parameter to be regularized, respectively
        :return: list of 2D tuples if regularization is used, otherwise an empty list
        """
        if not self.use_regularization: return []

        regularization_indicators = []
        for i, mod in enumerate(self.net):

            # indicator for last layer
            if i == self.last_layer_index:
                if self.sparsify_last_layer:
                    param_name = "weights" if self.sparsity_greater_than_zero else "weight"
                    regularization_tuple = (True, param_name)
                else:
                    regularization_tuple = (False, None)

                regularization_indicators.append(regularization_tuple)
                continue

            # indicator for non-parametric layers
            is_non_parametric = not hasattr(mod, "weight")
            if is_non_parametric:
                regularization_indicators.append((False, None))
                continue

            if isinstance(mod, torch.nn.Linear):
                param_name = "weight"
            elif isinstance(mod, SparseLinear):
                param_name = "weights"
            else:
                raise ValueError("Don't know how to handle this type of module: {0}".format(mod))

            regularization_indicators.append((True, param_name))

        return regularization_indicators

    def _initialize_summaries(self):
        """
        Initializes the summaries for the experiment
        """
        total_checkpoints = self.num_images_per_epoch * self.num_epochs // self.checkpoint
        self.results_dict["train_loss_per_checkpoint"] = torch.zeros(total_checkpoints, device=self.device,
                                                                     dtype=torch.float32)
        self.results_dict["train_accuracy_per_checkpoint"] = torch.zeros(total_checkpoints, device=self.device,
                                                                         dtype=torch.float32)
        self.results_dict["epoch_runtime"] = torch.zeros(self.num_epochs, device=self.device, dtype=torch.float32)
        # test summaries
        if not self.permute_inputs:
            self.results_dict["test_loss_per_epoch"] = torch.zeros(self.num_epochs, device=self.device,
                                                                        dtype=torch.float32)
            self.results_dict["test_accuracy_per_epoch"] = torch.zeros(self.num_epochs, device=self.device,
                                                                            dtype=torch.float32)
            self.results_dict["test_evaluation_runtime"] = torch.zeros(self.num_epochs, device=self.device,
                                                                       dtype=torch.float32)
        if self.l1_factor > 0.0:
            self.results_dict["train_reg_loss_per_checkpoint"] = torch.zeros(total_checkpoints, device=self.device,
                                                                             dtype=torch.float32)

        if self.sparsity_greater_than_zero:
            total_topology_updates = self.num_images_per_epoch * self.num_epochs // self.topology_update_frequency
            num_layers = self.num_layers if not self.sparsify_last_layer else self.num_layers + 1
            self.results_dict["num_units_pruned"] = torch.zeros((num_layers, total_topology_updates),
                                                                dtype=torch.float32)

    def _get_transformations(self):
        """
        Initializes all the non-stationary transformations applied to the data on each epoch
        :return: list of transformations
        """
        transformations = []

        current_scale = - self.scale_increase
        current_rotation = - self.rotation_increase
        grayscale_image = True

        for current_transformation_number in range(self.num_epochs):

            temp_transformations = []

            # horizontal flip
            if (current_transformation_number % 2) == 1:
                temp_transformations.append(RandomHorizontalFlip(p=1.0))

            if (current_transformation_number % 50) == 0:
                grayscale_image = not grayscale_image
            if grayscale_image:
                temp_transformations.append(GrayScale(num_output_channels=3, swap_colors=True))

            # random eraser
            if (current_transformation_number % 100) == 0:
                current_scale = round(current_scale + self.scale_increase, 3)
            if current_scale > 0.0:
                temp_transformations.append(RandomErasing(scale=(current_scale, round(current_scale + 0.01, 3)),
                                                          ratio=(1,2), value=(0,0,0), swap_colors=True))

            # random rotation
            if (current_transformation_number % 50) == 0:
                current_rotation = - self.rotation_increase
            if (current_transformation_number % 2) == 0:
                current_rotation = round(current_rotation + self.rotation_increase, 1)
            if current_rotation > 0.0:
                degrees = (round(current_rotation - 0.1,1), current_rotation)
                temp_transformations.append(RandomRotator(degrees))

            transformations.append(temp_transformations)

        if self.reverse_transformation_order:
            transformations.reverse()

        # for debugging:
        # for current_transformation_number, temp_transformations in enumerate(transformations):
        #     print("Transformation Number: {0}".format(current_transformation_number))
        #     for trans in temp_transformations:
        #         if isinstance(trans, RandomHorizontalFlip):
        #             print("\tHorizontal Flip")
        #         if isinstance(trans, GrayScale):
        #             print("\tGray Scale Image")
        #         if isinstance(trans, RandomErasing):
        #             print("\tErase: {0}".format(trans.eraser.scale))
        #         if isinstance(trans, RandomRotator):
        #             print("\tRotate: {0}".format(trans.rotator.degrees))

        return transformations

    # ----------------------------- For storing summaries ----------------------------- #
    def _store_training_summaries(self):

        # store train data
        self.results_dict["train_loss_per_checkpoint"][self.current_ckpt] += self.running_loss / self.checkpoint
        self.results_dict["train_accuracy_per_checkpoint"][self.current_ckpt] += self.running_accuracy / self.checkpoint

        if self.l1_factor > 0.0:
            self.results_dict["train_reg_loss_per_checkpoint"][self.current_ckpt] += self.running_regularized_loss / self.checkpoint

        self._print("\t\tOnline accuracy: {0:.2f}".format(self.running_accuracy / self.checkpoint))
        self.running_loss *= 0.0
        self.running_accuracy *= 0.0
        self.running_regularized_loss *= 0.0
        self.current_ckpt += 1

    def _store_test_summaries(self, test_data, epoch_number: int, epoch_runtime: float):

        self.results_dict["epoch_runtime"][epoch_number] += torch.tensor(epoch_runtime, dtype=torch.float32)
        # store test data
        if not self.permute_inputs:
            self.net.eval()
            test_evaluation_start_time = time.perf_counter()
            test_loss, test_accuracy = self.evaluate_network(test_data)
            test_evaluation_end_time = time.perf_counter()
            self.net.train()
            evaluation_run_time = test_evaluation_end_time - test_evaluation_start_time
            self.results_dict["test_evaluation_runtime"][epoch_number] += torch.tensor(evaluation_run_time, dtype=torch.float32)
            self.results_dict["test_loss_per_epoch"][epoch_number] += test_loss
            self.results_dict["test_accuracy_per_epoch"][epoch_number] += test_accuracy
            self._print("\t\tTest accuracy: {0:.4f}".format(test_accuracy))
            self._print("\t\tTest evaluation run time in seconds: {0:.4f}".format(evaluation_run_time))
        self._print("\t\tEpoch run time in seconds: {0:.4f}".format(epoch_runtime))

    def evaluate_network(self, test_data: DataLoader):
        """
        Evaluates the network on the test data
        :param test_data: a pytorch DataLoader object
        :return: (torch.Tensor) test loss, (torch.Tensor) test accuracy
        """

        avg_loss = 0.0
        avg_acc = 0.0
        with torch.no_grad():
            for _, sample in enumerate(test_data):
                images = sample["image"] if self.is_conv else sample["image"].reshape(self.batch_size, self.flat_image_dims)
                test_labels = sample["label"].to(self.device)
                test_outputs = self.net.forward(images.to(self.device))

                avg_loss += self.loss(test_outputs, test_labels)
                avg_acc += torch.mean((test_outputs.argmax(axis=1) == test_labels.argmax(axis=1)).to(torch.float32))

        return avg_loss / self.num_test_batches, avg_acc / self.num_test_batches

    def _store_pruning_summaries(self, num_different: list):
        """
        stores the summaries about each topology update
        :param num_different: (list) number of different connections in each layer of the new network
        :param proportion_different: (list) proportion of different connections in each layer of the new network
        """
        for i in range(len(num_different)):
            self.results_dict["num_units_pruned"][i, self.current_topology_update] += num_different[i].cpu()

    # --------------------------- For running the experiment --------------------------- #
    def run(self):
        # load data
        training_data, training_dataloader = self.get_data(train=True, return_data_loader=True)
        test_data, test_dataloader = self.get_data(train=False, return_data_loader=True)

        # train network
        self.train(train_dataloader=training_dataloader, test_dataloader=test_dataloader,
                   test_data=test_data, training_data=training_data)

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
        mnist_data = CifarDataSet(root_dir=self.data_path,
                                  train=train,
                                  cifar_type=10,
                                  device=None,
                                  image_normalization="minus-one-to-one",
                                  label_preprocessing="one-hot",
                                  use_torch=True)
        if return_data_loader:
            num_workers = 1 if self.device.type == "cpu" else 12
            dataloader = DataLoader(mnist_data, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
            return mnist_data, dataloader

        return mnist_data

    def train(self, train_dataloader: DataLoader, test_dataloader: DataLoader, test_data: CifarDataSet,
              training_data: CifarDataSet):

        for e in range(self.num_epochs):
            self._print("\tEpoch number: {0}".format(e + 1))
            self._transform_data(training_data, test_data)
            epoch_start_time = time.perf_counter()
            for step_number, sample in enumerate(train_dataloader):
                # sample observationa and target
                image = sample["image"].reshape(self.batch_size, np.prod(self.image_dims)).to(self.device)
                label = sample["label"].to(self.device)

                # reset gradients
                for param in self.net.parameters(): param.grad = None  # apparently faster than optim.zero_grad()

                # compute prediction and loss
                predictions = self.net.forward(image)
                current_reg_loss = self.loss(predictions, label)
                current_loss = current_reg_loss.detach().clone()

                # backpropagate and update weights
                current_reg_loss.backward()
                self._apply_regularization()
                self.optim.step()

                # store summaries
                current_accuracy = torch.mean((predictions.argmax(axis=1) == label.argmax(axis=1)).to(torch.float32))
                self.running_loss += current_loss
                self.running_accuracy += current_accuracy.detach()
                if self.use_l1_regularization:
                    self.running_regularized_loss += current_reg_loss.detach()
                if (step_number + 1) % self.checkpoint == 0:
                    self._print("\t\tStep Number: {0}".format(step_number + 1))
                    self._store_training_summaries()

                # update topology
                self._inject_noise_and_prune(step=step_number + 1)

            epoch_end_time = time.perf_counter()
            self._store_test_summaries(test_dataloader, epoch_number=e, epoch_runtime=epoch_end_time - epoch_start_time)

            if self.permute_inputs:
                training_data.set_transformation(Permute(np.random.permutation(np.arange(np.prod(self.image_dims)))))

    def _transform_data(self, training_data: CifarDataSet, test_data: CifarDataSet):
        """
        Applies a random transformation to the training data
        :param training_data: CIFAR-10 train data
        :param test_data: CIFAR-10 test data
        :return: None, but transforms the data in the training data set
        """

        transformations = self.transformations[self.num_transformations]
        if len(transformations) > 0:
            new_transformation = transforms.Compose(transformations)
            training_data.set_transformation(new_transformation)
            test_data.set_transformation(new_transformation)

        for trans in transformations:
            if isinstance(trans, RandomHorizontalFlip):
                print("\tHorizontal Flip")
            if isinstance(trans, GrayScale):
                print("\tGray Scale Image")
            if isinstance(trans, RandomErasing):
                print("\tErase: {0}\t{1}".format(trans.eraser.scale, trans.eraser.ratio))
            if isinstance(trans, RandomRotator):
                print("\tRotate: {0}".format(trans.rotator.degrees))

        self.num_transformations += 1

    # def _transform_data(self, training_data: CifarDataSet):
    #     """
    #     Applies a random transformation to the training data
    #     :param training_data: an instance of CifarDataSet from mlproj-manager
    #     :return: None, but transforms the data in the training data set
    #     """
    #
    #     new_transformation = transforms.Compose(
    #         [RandomVerticalFlip(p=0.5), RandomHorizontalFlip(p=0.5),
    #          RandomGaussianNoise(mean=0.0, stddev=0.05),
    #          RandomRotator(degrees=(0, 10))]
    #     )
    #     training_data.set_transformation(new_transformation)
    #     return
    #
    #     if self.num_transformations % 4 == 0:   # multiple of 4
    #         vertical_prob, horizontal_prob = (0.0, 0.0)
    #     elif self.num_transformations % 4 == 1:
    #         vertical_prob, horizontal_prob = (0.0, 1.0)
    #     elif self.num_transformations % 4 == 2:
    #         vertical_prob, horizontal_prob = (1.0, 0.0)
    #     else:
    #         vertical_prob, horizontal_prob = (1.0, 1.0)
    #
    #     transformations = []
    #     if horizontal_prob > 0.0:
    #         transformations.append(RandomHorizontalFlip(p=horizontal_prob))
    #     if vertical_prob > 0.0:
    #         transformations.append(RandomVerticalFlip(p=vertical_prob))
    #
    #     if self.num_transformations % 4 == 0:
    #         self.current_rotation += self.rotation_increase
    #         if self.current_rotation > 87.5:
    #             self.current_rotation = - self.rotation_increase
    #     if self.current_rotation > 0:
    #         degrees = (self.current_rotation - self.rotation_increase, self.current_rotation)
    #         transformations.append(RandomRotator(degrees=degrees))
    #
    #     # if self.num_transformations % 4 == 0:
    #     #     self.gaussian_noise_current_stddev += self.gaussian_noise_stddev_increase
    #     #     if self.gaussian_noise_current_stddev > 0.25:
    #     #         self.gaussian_noise_current_stddev = - self.gaussian_noise_stddev_increase
    #     # if self.gaussian_noise_current_stddev > 0:
    #     #     transformations.append(RandomGaussianNoise(mean=0, stddev=self.gaussian_noise_current_stddev))
    #
    #     if self.num_transformations % 100 == 0:
    #         self.scale += self.scale_increase
    #     if self.scale > 0.0:
    #         transformations.append(RandomErasing(scale=(self.scale, self.scale + 0.01), ratio=(1, 1), value=(0, 0, 0),
    #                                              swap_colors=True))
    #
    #     if len(transformations) > 0:
    #         new_transformation = transforms.Compose(transformations)
    #         training_data.set_transformation(new_transformation)
    #     print(transformations)
    #     self.num_transformations += 1

    def _apply_regularization(self):
        """
        Applies regularization to each layer of the network
        """
        if not self.use_regularization:
            return

        for mod_idx, mod in enumerate(self.net):
            apply_reg, param_name = self.regularization_indicators[mod_idx]
            if apply_reg:
                apply_regularization_to_module(module=mod, parameter_name=param_name, l1_factor=self.l1_factor, l2_factor=self.l2_factor)

    def _inject_noise_and_prune(self, step: int):
        """
        Updates the topology of the network by doing local or global pruning
        :param step: the current training step
        :return: None, but stores summaries relevant to the pruning step
        """

        # check if topology should be updated
        matches_frequency = ((step % self.topology_update_frequency) == 0)
        time_to_update_topology = matches_frequency and self.sparsity_greater_than_zero
        if not time_to_update_topology:
            return

        # do global or local pruning and update topology with new random connections
        if self.global_pruning:
            # Haven't figured out a way to implement this efficiently
            raise NotImplemented
        else:
            num_different_per_layer = self._inject_noise_and_prune_local()

        self._store_pruning_summaries(num_different_per_layer)
        self.current_topology_update += 1

    def _inject_noise_and_prune_local(self):
        """
        Update the topology of the network by replacing weights whose magnitude is lower than the magnitude of randomly
        initialized weights using the initialization distribution. Specifically, the topology update follows these
        steps:
            1. For each sparse layer, retrieves the dense mask corresponding to the sparse weights.
            2. Creates a new weight matrix where the weights corresponding to the ones in the mask are left the same and
               the weights corresponding to the zeros in the mask are randomly initialized using the initialization
               distribution.
            3. Creates a new mask where only the top-k weights, according to weight magnitude, are assigned a one.
            4. Creates a masked version of the new weights.
            5. The masked weights are turn into a sparse matrix and then the values and indices of the weights are
               copied onto the current sparse module.
        Dense modules are left unchanged as well as the bias terms of all the modules.
        :return: None, but modifies the weight values and indices of the sparse modules in the network
        """

        num_different_per_layer = []
        for i, mod in enumerate(self.net):

            # if the last layer is not sparse, then continue
            is_last_layer = (i == self.last_layer_index)
            if is_last_layer and (not self.sparsify_last_layer): continue

            # skip layers that don't have any parameters
            is_non_parametric = not hasattr(mod, "weight")
            if is_non_parametric: continue

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
                    plt.ylim((0.2,0.7))
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
        "stepsize": 0.01,   # 0.01 for mnist, 0.005 for cifar 10
        "l1_factor": 0.0,    # 0.0000001 for cifar 10
        "l2_factor": 0.0,
        "topology_update_frequency": 20,
        "sparsity_level": 0.0,
        "global_pruning": False,
        "data_path": os.path.join(file_path, "data"),
        "num_epochs": 1000,
        "num_layers": 3,
        "num_hidden": 4000,
        "activation_function": "relu",
        "permute_inputs": False,
        "sparsify_last_layer": False,
        "reverse_transformation_order": True,
        "plot": False
    }

    print(experiment_parameters)
    relevant_parameters = ["topology_update_frequency", "sparsity_level", "num_epochs", "num_layers", "num_hidden",
                           "sparsify_last_layer", "reverse_transformation_order"]
    results_dir_name = ""
    for relevant_param in relevant_parameters:
        results_dir_name += "_" + relevant_param + "-" + str(experiment_parameters[relevant_param])

    initial_time = time.perf_counter()
    exp = DynamicSparseCIFARExperiment(experiment_parameters,
                                       results_dir=os.path.join(file_path, "results", results_dir_name),
                                       run_index=0,
                                       verbose=True)
    exp.run()
    exp.store_results()
    final_time = time.perf_counter()
    print("The running time in minutes is: {0:.2f}".format((final_time - initial_time) / 60))


if __name__ == "__main__":
    main()
