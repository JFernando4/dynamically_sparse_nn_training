"""

Minimal implementation of incremental CIFAR-100 experiment using continual backpropagation

"""

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

from src import initialize_layer_norm_module, ResGnT, ResNet, initialize_vit
from src.utils import get_cifar_data, compute_accuracy_from_batch, parse_terminal_arguments
from src.networks.torchvision_modified_vit import VisionTransformer
from src.utils.cifar100_experiment_utils import IncrementalCIFARExperiment, save_model_parameters


class ResNetIncrementalCIFARExperiment(IncrementalCIFARExperiment):

    def __init__(self, exp_params: dict, results_dir: str, run_index: int, verbose=True):
        super().__init__(exp_params, results_dir, run_index, verbose)

        """ Experiment parameters """
        # optimization parameters
        self.stepsize = exp_params["stepsize"]
        self.weight_decay = exp_params["weight_decay"]
        self.rescaled_wd = access_dict(exp_params, "rescaled_wd", default=False, val_type=bool)
        self.momentum = exp_params["momentum"]
        self.use_lr_schedule = access_dict(exp_params, "use_lr_schedule", default=False, val_type=bool)
        self.dropout_prob = access_dict(exp_params, "dropout_prob", default=0.05, val_type=float)
        self.reset_layer_norm = access_dict(exp_params, "reset_layer_norm", default=False, val_type=bool)

        # cbp parameters
        self.replacement_rate = access_dict(exp_params, "replacement_rate", default=0.0, val_type=float)
        assert self.replacement_rate > 0.0
        self.utility_function = access_dict(exp_params, "utility_function", default="weight", val_type=str,
                                            choices=["weight", "contribution"])
        self.maturity_threshold = access_dict(exp_params, "maturity_threshold", default=0, val_type=int)
        assert self.maturity_threshold > 0

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
            attention_dropout=self.dropout_prob
        )
        initialize_vit(self.net)
        self.net.to(self.device)

        for n, p in self.net.named_parameters():
            print(n)

        # initialize optimizer and loss
        self.optim = torch.optim.SGD(self.net.parameters(), lr=self.stepsize, momentum=self.momentum,
                                     weight_decay=self.weight_decay)
        self.lr_scheduler = None
        self.loss = torch.nn.CrossEntropyLoss(reduction="mean")

        # move network to device
        self.current_epoch = 0
        self.current_minibatch = 0

        # for cbp
        self.resgnt = None
        # self.resgnt = ResGnT(net=self.net,
        #                      hidden_activation="gelu",
        #                      replacement_rate=self.replacement_rate,
        #                      decay_rate=0.99,
        #                      util_type=self.utility_function,
        #                      maturity_threshold=self.maturity_threshold,
        #                      device=self.device)

        """ For data partitioning """
        self.class_increase_frequency = 200

        """ For creating experiment checkpoints """
        self.checkpoint_save_frequency = self.class_increase_frequency  # save every time a new class is added

        """ For summaries """
        self._initialize_summaries()

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
            "partial_results": partial_results,
            "resgnt": self.resgnt
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
        self.current_num_classes = checkpoint["current_num_classes"]
        self.all_classes = checkpoint["all_classes"]
        self.current_running_avg_step = checkpoint["current_running_avg_step"]
        self.resgnt = checkpoint["resgnt"]

        partial_results = checkpoint["partial_results"]
        for k, v in self.results_dict.items():
            self.results_dict[k] = partial_results[k] if not isinstance(partial_results[k], torch.Tensor) else partial_results[k].to(self.device)

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
        if self.use_lr_schedule:
            self.lr_scheduler = self.get_lr_scheduler(steps_per_epoch=len(train_dataloader))
        save_model_parameters(self.results_dir, self.run_index, self.current_epoch, self.net)

        for e in range(self.current_epoch, self.num_epochs):
            self._print("\tEpoch number: {0}".format(e + 1))

            epoch_start_time = time.perf_counter()
            for step_number, sample in enumerate(train_dataloader):
                # sample observationa and target
                image = sample["image"].to(self.device)
                label = sample["label"].to(self.device)

                # reset gradients
                for param in self.net.parameters(): param.grad = None   # apparently faster than optim.zero_grad()

                # compute prediction and loss
                current_features = []
                predictions = self.net.forward(image, current_features)[:, self.all_classes[:self.current_num_classes]]
                current_loss = self.loss(predictions, label)
                detached_loss = current_loss.detach().clone()

                # backpropagate and update weights
                current_loss.backward()
                self.optim.step()
                if self.use_lr_schedule:
                    self.lr_scheduler.step()
                    if self.lr_scheduler.get_last_lr()[0] > 0.0 and not self.rescaled_wd:
                        self.optim.param_groups[0]['weight_decay'] = self.weight_decay / self.lr_scheduler.get_last_lr()[0]

                # CBP step
                # self.resgnt.gen_and_test(current_features)

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
            classes_extended = self.extend_classes(training_data, test_data, val_data)
            if classes_extended and self.use_lr_schedule:
                self.lr_scheduler = self.get_lr_scheduler(steps_per_epoch=len(train_dataloader))

            if self.current_epoch % self.checkpoint_save_frequency == 0:
                self.save_experiment_checkpoint()

    def get_lr_scheduler(self, steps_per_epoch: int):
        """ Returns lr scheduler used in the original Vision Transformers paper """
        scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optim, max_lr=self.stepsize, anneal_strategy="linear",
                                                        epochs=self.class_increase_frequency,
                                                        steps_per_epoch=steps_per_epoch)
        if not self.rescaled_wd:
            self.optim.param_groups[0]['weight_decay'] = self.weight_decay / scheduler.get_last_lr()[0]
        return scheduler

    def post_class_increase_processing(self):
        """ Performs optional operations after the number of classes has been increased """
        if self.reset_layer_norm:
            self.net.apply(initialize_layer_norm_module)

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
