# built-in libraries
import time
import os
import argparse
import pickle

# third party libraries
import torch
from torch.utils.data import DataLoader
import numpy as np
import wandb

# from ml project manager
from mlproj_manager.experiments import Experiment
from mlproj_manager.problems import MnistDataSet
from mlproj_manager.util import access_dict, Permute, get_random_seeds, turn_off_debugging_processes
from mlproj_manager.util.neural_networks import init_weights_kaiming

# from src
from src.sparsity_funcs import *
from src.utils import apply_regularization_to_sequential_net


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
        self.stepsize = exp_params["stepsize"]
        self.l1_factor = exp_params["l1_factor"]
        self.l2_factor = exp_params["l2_factor"]
        self.use_regularization = (self.l2_factor > 0.0) or (self.l1_factor > 0.0)
        self.data_path = exp_params["data_path"]
        self.num_epochs = exp_params["num_epochs"]      # number of training epochs
        self.num_layers = exp_params["num_layers"]      # number of hidden layers
        self.num_hidden = exp_params["num_hidden"]      # number of hidden units per hidden layer
        self.dst_algorithm = access_dict(exp_params, "algorithm", default="set",
                                         choices=["set", "dense", "static_sparse"])
        self.sparsity = access_dict(exp_params, "sparsity", default=0.0, val_type=float)
        self.results_dir = results_dir

        """ Training constants """
        self.batch_size = 1
        self.num_classes = 10
        self.num_inputs = 784
        self.num_images_per_epoch = 60000

        """ Network set up """
        self.net, self.masks = self.initialize_network()

        # initialize optimizer
        self.optim = torch.optim.SGD(self.net.parameters(), lr=self.stepsize)

        # define loss function
        self.loss = torch.nn.CrossEntropyLoss(reduction="mean")

        # move network to device
        self.net.to(self.device)

        """ Experiment Summaries """
        self.running_avg_window = 100
        self.current_running_avg_step, self.running_loss, self.running_accuracy, self.current_epoch = (0, 0.0, 0.0, 0)
        self.results_dict = {}
        total_checkpoints = self.num_images_per_epoch * self.num_epochs // (self.running_avg_window * self.batch_size)
        self.results_dict["train_loss_per_checkpoint"] = torch.zeros(total_checkpoints, device=self.device,
                                                                     dtype=torch.float32)
        self.results_dict["train_accuracy_per_checkpoint"] = torch.zeros(total_checkpoints, device=self.device,
                                                                         dtype=torch.float32)
        """ Wandb set up """
        self.wandb_run = self.initialize_wandb(exp_params)

        """ For creating experiment checkpoints """
        self.current_epoch = 0
        self.experiment_checkpoints_dir_path = os.path.join(self.results_dir, "experiment_checkpoints")
        self.checkpoint_identifier_name = "current_epoch"
        self.checkpoint_save_frequency = 1     # create running_avg_window every 10 epochs
        self.load_experiment_checkpoint()

    # ----------------------------- For initializing the experiment ----------------------------- #
    def initialize_wandb(self, exp_params: dict):
        """ Initializes the weights and biases session """

        # set wandb directory and mode
        os.environ["WANDB_DIR"] = access_dict(exp_params, "wandb_dir", default=self.results_dir, val_type=str)
        wandb_mode = access_dict(exp_params, "wandb_mode", default="offline", val_type=str, choices=["online", "offline", "disabled"])

        # retrieve tags for the current run
        wandb_tags = access_dict(exp_params, "wandb_tags", default="", val_type=str)  # comma separated string of tags
        tag_list = None if wandb_tags == "" else wandb_tags.split(",")

        # set console log file
        slurm_job_id = "0" if "SLURM_JOBID" not in os.environ else os.environ["SLURM_JOBID"]
        exp_params["slurm_job_id"] = slurm_job_id

        run_id = self.get_wandb_id()
        run_name = "{0}_index-{1}".format(os.path.basename(self.results_dir), self.run_index)
        return wandb.init(project="dstlop", entity="dst-lop", mode=wandb_mode, config=exp_params, tags=tag_list,
                          name=run_name, id=run_id)

    def get_wandb_id(self):
        """ Generates and stores a wandb id or loads it if one is already available """

        wandb_id_dir = os.path.join(self.results_dir, "wandb_ids")
        os.makedirs(wandb_id_dir, exist_ok=True)

        wandb_id_filepath = os.path.join(wandb_id_dir, "index-{0}.p".format(self.run_index))

        # an id was already stored
        if os.path.isfile(wandb_id_filepath):
            with open(wandb_id_filepath, mode="rb") as id_file:
                run_id = pickle.load(id_file)
            return run_id

        # generate a new id
        run_id = wandb.util.generate_id()
        number_of_attempts = 10
        for i in range(number_of_attempts):
            try:
                with open(wandb_id_filepath, mode="wb") as id_file:
                    pickle.dump(run_id, id_file)
                with open(wandb_id_filepath, mode="rb") as id_file:
                    pickle.load(id_file)
                break
            except ValueError:
                print("Something went wrong on attempt {0} when loading the job id.".format(i + 1))
            print("Couldn't store the run id. Checkpointing won't be enable.")

        return run_id

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
            net.append(torch.nn.ReLU())
            in_features = out_features
        # output layer
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
            "epoch_number": self.current_epoch,
            "current_running_avg_step": self.current_running_avg_step,
            "current_running_averages": (self.running_accuracy, self.running_loss),
            "partial_results": partial_results
        }

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
        self.current_epoch = checkpoint["epoch_number"]
        self.current_running_avg_step = checkpoint["current_running_avg_step"]
        self.running_accuracy, self.running_loss = checkpoint["current_running_averages"]

        if self.device.type == "cuda":
            torch.cuda.set_rng_state(checkpoint["cuda_rng_state"])

        partial_results = checkpoint["partial_results"]
        # log partial results to wandb
        for i in range(self.current_running_avg_step):
            temp_results = {
                "train_loss_per_checkpoint": partial_results["train_loss_per_checkpoint"][i],
                "train_accuracy_per_checkpoint":partial_results["train_accuracy_per_checkpoint"][i]
            }
            wandb.log(temp_results, step=(self.current_running_avg_step + 1) * self.running_avg_window)

        # store partial results
        for k, v in self.results_dict.items():
            self.results_dict[k] = partial_results[k] if not isinstance(partial_results[k], torch.Tensor) else \
            partial_results[k].to(self.device)

    # ----------------------------- For storing summaries ----------------------------- #
    def _store_training_summaries(self):

        current_results = {
            "train_loss_per_checkpoint": self.running_loss / self.running_avg_window,
            "train_accuracy_per_checkpoint": self.running_accuracy / self.running_avg_window
        }
        wandb.log(data=current_results, step=(self.current_running_avg_step + 1) * self.running_avg_window)

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

        # for e in range(self.num_epochs):
        while self.current_epoch < self.num_epochs:

            training_data.set_transformation(Permute(np.random.permutation(self.num_inputs)))  # apply new permutation
            print("\tEpoch number: {0}".format(self.current_epoch + 1))

            for i, sample in enumerate(mnist_data_loader):
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
                if self.use_regularization: apply_regularization_to_sequential_net(self.net)
                self.optim.step()
                if self.sparsity > 0.0: apply_weight_masks(self.masks)

                # store summaries
                current_accuracy = torch.mean((predictions.argmax(axis=1) == label.argmax(axis=1)).to(torch.float32))
                self.running_loss += current_loss
                self.running_accuracy += current_accuracy.detach()
                if (i + 1) % self.running_avg_window == 0:
                    self._print("\t\tStep Number: {0}".format(i + 1))
                    self._store_training_summaries()

            self.current_epoch += 1
            if self.current_epoch % self.checkpoint_save_frequency == 0:    # checkpoint experiment
                self.save_experiment_checkpoint()


def parse_args():
    file_path = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, default=0,
                        help="Run index; a unique random seed is assigned to each different index.")
    parser.add_argument('--stepsize', type=float, default=0.001)
    parser.add_argument('--l1_factor', type=float, default=0.0)
    parser.add_argument('--l2_factor', type=float, default=0.0)
    parser.add_argument('--data_path', type=str, default=os.path.join(file_path, "data"))
    parser.add_argument('--results_dir', type=str, default=os.path.join(file_path, "results"))
    parser.add_argument('--num_epochs', type=int, default=5 )
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--num_hidden', type=int, default=100)
    parser.add_argument('--algorithm', type=str, default='static_sparse',
                        help="Algorithm to use for training.", choices=["set", "rigl", "static_sparse", "dense"])
    parser.add_argument('--sparsity', type=int, default=0.8)
    parser.add_argument('--reinit_method', type=str, default='zero', choices=['zero', 'kaiming_normal'],
                        help="How to reinitialize the weights that are regrown.")
    parser.add_argument('--wandb_dir', type=str, default=os.path.join(file_path, "results"))
    parser.add_argument('--wandb_mode', type=str, default='offline', choices=['online', 'offline', 'disabled'])
    parser.add_argument('--wandb_tags', type=str, default="test", help="String of comma separated tags.")
    parser.add_argument('--verbose', type=bool, default=True)
    args = parser.parse_args()
    return args


def main():
    """
    This is a quick demonstration of how to run the experiments. For a more systematic run, use the mlproj_manager
    scheduler.
    """
    args = parse_args()
    experiment_parameters = vars(args)

    print(experiment_parameters)

    relevant_parameters = ["num_epochs", "num_layers", "num_hidden", "algorithm", "sparsity", "stepsize",
                           "l1_factor", "l2_factor"]
    results_dir_name = "{0}-{1}".format(relevant_parameters[0], experiment_parameters[relevant_parameters[0]])
    for relevant_param in relevant_parameters[1:]:
        results_dir_name += "_" + relevant_param + "-" + str(experiment_parameters[relevant_param])

    initial_time = time.perf_counter()
    exp = PermutedMNISTExperiment(experiment_parameters,
                                  results_dir=os.path.join(experiment_parameters["results_dir"], results_dir_name),
                                  run_index=experiment_parameters["index"],
                                  verbose=experiment_parameters["verbose"])
    exp.run()
    final_time = time.perf_counter()
    print("The running time in minutes is: {0:.2f}".format((final_time - initial_time) / 60))


if __name__ == "__main__":
    main()
