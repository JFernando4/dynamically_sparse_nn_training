# built-in libraries
import time
import os
import pickle

# third party libraries
import torch
from torch.utils.data import DataLoader
import numpy as np

# from ml project manager
from mlproj_manager.problems import CifarDataSet
from mlproj_manager.util import access_dict

from src import kaiming_init_resnet_module, build_resnet18, ResGnT, ResNet
from src.cbpw_functions import initialize_weight_dict, update_weights
from src.plasticity_functions import inject_noise
from src.utils import get_cifar_data, compute_accuracy_from_batch, parse_terminal_arguments
from src.utils.cifar100_experiment_utils import IncrementalCIFARExperiment, save_model_parameters


class ResNetIncrementalCIFARExperiment(IncrementalCIFARExperiment):

    def __init__(self, exp_params: dict, results_dir: str, run_index: int, verbose=True):
        super().__init__(exp_params, results_dir, run_index, verbose)

        """ Experiment parameters """
        # optimization parameters
        self.stepsize = exp_params["stepsize"]
        self.weight_decay = exp_params["weight_decay"]
        self.momentum = exp_params["momentum"]
        self.use_lr_schedule = access_dict(exp_params, "use_lr_schedule", default=False, val_type=bool)

        # network resetting parameters
        self.reset_head = access_dict(exp_params, "reset_head", default=False, val_type=bool)
        self.reset_network = access_dict(exp_params, "reset_network", default=False, val_type=bool)
        if self.reset_head and self.reset_network:
            print(Warning("Resetting the whole network supersedes resetting the head of the network. There's no need to set both to True."))

        # cbp parameters
        self.use_cbp = access_dict(exp_params, "use_cbp", default=False, val_type=bool)
        self.replacement_rate = access_dict(exp_params, "replacement_rate", default=0.0, val_type=float)
        assert (not self.use_cbp) or (self.replacement_rate > 0.0)
        self.utility_function = access_dict(exp_params, "utility_function", default="weight", val_type=str,
                                            choices=["weight", "contribution"])
        self.maturity_threshold = access_dict(exp_params, "maturity_threshold", default=0, val_type=int)
        assert (not self.use_cbp) or (self.maturity_threshold > 0)

        # CBPw parameters
        self.use_cbpw = access_dict(exp_params, "use_cbpw", default=False, val_type=bool)
        self.topology_update_freq = access_dict(exp_params, "topology_update_freq", default=0, val_type=int)
        pruning_functions_names = ["none", "magnitude", "redo", "gf", "hess_approx"]
        self.prune_method = access_dict(exp_params, "prune_method", default="none", val_type=str, choices=pruning_functions_names)
        grow_methods = ["none", "pm_min", "xavier_normal", "zero"]
        self.grow_method = access_dict(exp_params, "grow_method", default="none", val_type=str, choices=grow_methods)
        assert not ((self.prune_method != "none" and self.grow_method == "none") or (self.prune_method == "none" and self.grow_method != "none"))
        self.drop_factor = access_dict(exp_params, "drop_factor", default=float, val_type=float)
        self.bn_cbpw = access_dict(exp_params, "bn_cbpw", default=False, val_type=bool)
        self.current_topology_update = 0

        # shrink and perturb parameters
        self.noise_std = access_dict(exp_params, "noise_std", default=0.0, val_type=float)
        self.perturb_weights_indicator = self.noise_std > 0.0

        """ Network set up """
        # initialize network
        self.net = build_resnet18(num_classes=self.num_classes, norm_layer=torch.nn.BatchNorm2d)
        self.net.apply(kaiming_init_resnet_module)
        self.net.to(self.device)

        # initializes weight dictionary for CBPw
        self.weight_dict = None
        if self.use_cbpw:
            self.weight_dict = initialize_weight_dict(self.net, "resnet", self.prune_method,
                                                      self.grow_method, self.drop_factor, include_bn=self.bn_cbpw)

        # initialize optimizer
        self.optim = torch.optim.SGD(self.net.parameters(), lr=self.stepsize, momentum=self.momentum,
                                     weight_decay=self.weight_decay)

        # define loss function
        self.loss = torch.nn.CrossEntropyLoss(reduction="mean")

        # move network to device
        self.current_epoch = 0
        self.current_minibatch = 0

        # for cbp
        self.resgnt = None
        if self.use_cbp:
            self.resgnt = ResGnT(net=self.net,
                                 hidden_activation="relu",
                                 replacement_rate=self.replacement_rate,
                                 decay_rate=0.99,
                                 util_type=self.utility_function,
                                 maturity_threshold=self.maturity_threshold,
                                 device=self.device)
        self.current_features = [] if self.use_cbp else None

        """ For data partitioning """
        self.class_increase_frequency = 200

        """ For creating experiment checkpoints """
        self.checkpoint_save_frequency = self.class_increase_frequency  # save every time a new class is added

        """ For summaries """
        self._initialize_summaries()

        # check placeholders have been replaced with correct values
        is_resnet =  isinstance(self.net, ResNet)
        is_sgd = isinstance(self.optim, torch.optim.SGD)
        is_positive_int = (self.checkpoint_save_frequency > 0) and (self.class_increase_frequency > 0)
        is_non_negative = (self.topology_update_freq >= 0) and (self.current_topology_update >= 0)
        is_bool = isinstance(self.use_cbpw, bool)
        is_correct_string = (self.prune_method in pruning_functions_names)
        assert is_resnet and is_sgd and is_positive_int and is_non_negative and is_bool and is_correct_string

    # ----------------------------- For saving and loading experiment checkpoints ----------------------------- #
    def get_experiment_checkpoint(self):
        """ Creates a dictionary with all the necessary information to pause and resume the experiment """

        partial_results = {}
        for k, v in self.results_dict.items():
            partial_results[k] = v if not isinstance(v, torch.Tensor) else v.cpu()

        checkpoint = {
            "model_weights": self.net.state_dict(),
            "optim_state": self.optim.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(),
            "epoch_number": self.current_epoch,
            "current_num_classes": self.current_num_classes,
            "all_classes": self.all_classes,
            "current_running_avg_step": self.current_running_avg_step,
            "partial_results": partial_results
        }

        if self.use_cbp:
            checkpoint["resgnt"] = self.resgnt

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
        self.current_num_classes = checkpoint["current_num_classes"]
        self.all_classes = checkpoint["all_classes"]
        self.current_running_avg_step = checkpoint["current_running_avg_step"]

        partial_results = checkpoint["partial_results"]
        for k, v in self.results_dict.items():
            self.results_dict[k] = partial_results[k] if not isinstance(partial_results[k], torch.Tensor) else partial_results[k].to(self.device)

        if self.use_cbp:
            self.resgnt = checkpoint["resgnt"]

        if self.use_cbpw:
            self.weight_dict = initialize_weight_dict(self.net, "resnet", self.prune_method,
                                                      self.grow_method, self.drop_factor, include_bn=self.bn_cbpw)

    # ------------------------------------- For running the experiment ------------------------------------- #
    def run(self):
        # load data
        training_data, training_dl = get_cifar_data(self.data_path, train=True, validation=False,
                                                    batch_size=self.batch_sizes["train"], num_workers=self.num_workers)
        val_data, val_dl = get_cifar_data(self.data_path, train=True, validation=True,
                                          batch_size=self.batch_sizes["validation"], num_workers=self.num_workers)
        test_data, test_dl = get_cifar_data(self.data_path, train=False, batch_size=self.batch_sizes["test"],
                                            num_workers=self.num_workers)

        self.load_experiment_checkpoint()
        # train network
        self.train(train_dataloader=training_dl, test_dataloader=test_dl, val_dataloader=val_dl, test_data=test_data,
                   training_data=training_data, val_data=val_data)

        # summaries stored in memory automatically if using mlproj_manager

    def train(self, train_dataloader: DataLoader, test_dataloader: DataLoader, val_dataloader: DataLoader,
              test_data: CifarDataSet, training_data: CifarDataSet, val_data: CifarDataSet):

        self.partition_data(training_data, test_data, val_data)
        save_model_parameters(self.results_dir, self.run_index, self.current_epoch, self.net)

        for e in range(self.current_epoch, self.num_epochs):
            self._print("\tEpoch number: {0}".format(e + 1))
            self.set_lr()

            epoch_start_time = time.perf_counter()
            for step_number, sample in enumerate(train_dataloader):
                # sample observationa and target
                image = sample["image"].to(self.device)
                label = sample["label"].to(self.device)

                # reset gradients
                for param in self.net.parameters(): param.grad = None   # apparently faster than optim.zero_grad()

                # compute prediction and loss
                current_features = [] if self.use_cbp else None
                predictions = self.net.forward(image, current_features)[:, self.all_classes[:self.current_num_classes]]
                current_loss = self.loss(predictions, label)
                detached_loss = current_loss.detach().clone()

                # backpropagate and update weights
                current_loss.backward()
                self.optim.step()
                if self.use_cbp: self.resgnt.gen_and_test(current_features)
                if self.perturb_weights_indicator: inject_noise(self.net, self.noise_std)
                if self.use_cbpw and (self.current_minibatch % self.topology_update_freq) == 0:
                    self._store_mask_update_summary(update_weights(self.weight_dict))

                # store summaries
                current_accuracy = compute_accuracy_from_batch(predictions, label)
                self.running_loss += detached_loss
                self.running_accuracy += current_accuracy.detach()
                if (step_number + 1) % self.running_avg_window == 0:
                    self._print("\t\tStep Number: {0}".format(step_number + 1))
                    self._store_training_summaries()

            epoch_runtime = time.perf_counter() - epoch_start_time
            self._store_test_summaries(test_dataloader, val_dataloader, epoch_number=e, epoch_runtime=epoch_runtime)

            self.current_epoch += 1
            self.extend_classes(training_data, test_data, val_data)

            if self.current_epoch % self.checkpoint_save_frequency == 0:
                self.save_experiment_checkpoint()

    def set_lr(self):
        """ Changes the learning rate of the optimizer according to the current epoch of the task """
        if not self.use_lr_schedule: return

        current_stepsize = None
        if (self.current_epoch % self.class_increase_frequency) == 0:
            current_stepsize = self.stepsize
        elif (self.current_epoch % self.class_increase_frequency) == 60:
            current_stepsize = round(self.stepsize * 0.2, 5)
        elif (self.current_epoch % self.class_increase_frequency) == 120:
            current_stepsize = round(self.stepsize * (0.2 ** 2), 5)
        elif (self.current_epoch % self.class_increase_frequency) == 160:
            current_stepsize = round(self.stepsize * (0.2 ** 3), 5)

        if current_stepsize is not None:
            for g in self.optim.param_groups:
                g['lr'] = current_stepsize
            self._print("\tCurrent stepsize: {0:.5f}".format(current_stepsize))

    def post_class_increase_processing(self):
        """ Performs optional operations after the number of classes has been increased """
        if self.reset_head:
            kaiming_init_resnet_module(self.net.fc)
        if self.reset_network:
            self.net.apply(kaiming_init_resnet_module)

def main():
    """
    Function for running the experiment from command line given a path to a json config file
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
    exp = ResNetIncrementalCIFARExperiment(experiment_parameters,
                                           results_dir=os.path.join(file_path, "results", results_dir_name),
                                           run_index=terminal_arguments.run_index,
                                           verbose=terminal_arguments.verbose)
    exp.run()
    exp.store_results()
    final_time = time.perf_counter()
    print("The running time in minutes is: {0:.2f}".format((final_time - initial_time) / 60))


if __name__ == "__main__":
    main()
