# built-in libraries
import time
import os
import pickle
import re
from copy import deepcopy

# third party libraries
import torch
from torch.utils.data import DataLoader
import numpy as np
from torchvision import transforms

# from ml project manager
from mlproj_manager.problems import CifarDataSet
from mlproj_manager.experiments import Experiment
from mlproj_manager.util import turn_off_debugging_processes, get_random_seeds, access_dict, init_weights_kaiming
from mlproj_manager.util.data_preprocessing_and_transformations import ToTensor, Normalize, RandomCrop, RandomHorizontalFlip, RandomRotator

from src import ResNet9, kaiming_init_resnet_module, build_resnet34, build_resnet18, build_resnet10


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
        self.stepsize = exp_params["stepsize"]
        self.weight_decay = exp_params["weight_decay"]
        self.momentum = exp_params["momentum"]
        self.data_path = exp_params["data_path"]
        self.num_epochs = access_dict(exp_params, "num_epochs", default=1, val_type=int)
        self.current_num_classes = access_dict(exp_params, "initial_num_classes", default=2, val_type=int)
        self.fixed_classes = access_dict(exp_params, "fixed_classes", default=True, val_type=bool)
        self.reset_head = access_dict(exp_params, "reset_head", default=False, val_type=bool)
        self.reset_network = access_dict(exp_params, "reset_network", default=False, val_type=bool)
        self.use_data_augmentation = access_dict(exp_params, "use_data_augmentation", default=False, val_type=bool)
        self.use_cifar100 = access_dict(exp_params, "use_cifar100", default=False, val_type=bool)
        self.use_lr_schedule = access_dict(exp_params, "use_lr_schedule", default=False, val_type=bool)
        if self.reset_head and self.reset_network:
            print(Warning("Resetting the whole network supersedes resetting the head of the network. There's no need to set both to True."))
        self.plot = access_dict(exp_params, key="plot", default=False)

        """ Training constants """
        self.batch_size = 100
        self.num_classes = 10 if not self.use_cifar100 else 100
        self.image_dims = (32, 32, 3)
        self.flat_image_dims = int(np.prod(self.image_dims))
        self.num_images_per_epoch = 50000
        self.num_test_samples = 10000

        """ Network set up """
        # initialize network
        # self.net = resnet18(num_classes=10, norm_layer=torch.nn.Identity)
        # self.net = ResNet9(in_channels=3, num_classes=self.num_classes, norm_function=torch.nn.BatchNorm2d)
        self.net = build_resnet18(num_classes=self.num_classes, norm_layer=torch.nn.BatchNorm2d)
        self.net.apply(kaiming_init_resnet_module)

        # initialize optimizer
        self.optim = torch.optim.SGD(self.net.parameters(), lr=self.stepsize, momentum=self.momentum,
                                     weight_decay=self.weight_decay)

        # define loss function
        self.loss = torch.nn.CrossEntropyLoss(reduction="mean")

        # move network to device
        self.net.to(self.device)
        self.current_epoch = 0

        """ For summaries """
        self.running_avg_window = 50
        self.current_running_avg_step, self.running_loss, self.running_accuracy = (0, 0.0, 0.0)
        self._initialize_summaries()

        """ For data partitioning """
        self.class_increase_frequency = 200
        self.all_classes = np.random.permutation(self.num_classes)
        self.best_accuracy = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        self.best_accuracy_model_parameters = {}

        """ For creating experiment checkpoints """
        self.experiment_checkpoints_dir_path = os.path.join(self.results_dir, "experiment_checkpoints")
        self.checkpoint_identifier_name = "current_epoch"
        self.checkpoint_save_frequency = self.class_increase_frequency  # save every time a new class is added
        self.delete_old_checkpoints = False

    # -------------------- Methods for initializing the experiment --------------------#
    def _initialize_summaries(self):
        """
        Initializes the summaries for the experiment
        """
        total_checkpoints = self.num_images_per_epoch * self.num_epochs // (self.running_avg_window * self.batch_size)
        self.results_dict["train_loss_per_checkpoint"] = torch.zeros(total_checkpoints, device=self.device,
                                                                     dtype=torch.float32)
        self.results_dict["train_accuracy_per_checkpoint"] = torch.zeros(total_checkpoints, device=self.device,
                                                                         dtype=torch.float32)
        self.results_dict["epoch_runtime"] = torch.zeros(self.num_epochs, device=self.device, dtype=torch.float32)
        # test summaries
        self.results_dict["test_loss_per_epoch"] = torch.zeros(self.num_epochs, device=self.device,
                                                                    dtype=torch.float32)
        self.results_dict["test_accuracy_per_epoch"] = torch.zeros(self.num_epochs, device=self.device,
                                                                        dtype=torch.float32)
        self.results_dict["test_evaluation_runtime"] = torch.zeros(self.num_epochs, device=self.device,
                                                                   dtype=torch.float32)

    def save_experiment_checkpoint(self):
        """
        Saves all the information necessary to resume the experiment with the same index. Specifically, it stores: the
        random states of torch and numpy, the current weights of the model, the current checkpoint, the current number
        of classes, and the randomized list of classes

        The function creates a file at self.training_checkpoint_path named:
            "index-$experiment_index_$(checkpoint_identifier_name)-$(checkpoint_identifier_value)

        The name should be an attribute of self defined in __init__ and the value should be an increasing sequence of
        integers where higher values correspond to latter steps of the experiment
        """

        os.makedirs(self.experiment_checkpoints_dir_path, exist_ok=True)
        checkpoint_identifier_value = getattr(self, self.checkpoint_identifier_name)

        if not isinstance(checkpoint_identifier_value, int):
            warning_message = "The checkpoint identifier should be an integer. Got {0} instead, which result in unexpected behaviour."
            print(Warning(warning_message.format(checkpoint_identifier_value.__class__)))

        file_name = "index-{0}_{1}-{2}.p".format(self.run_index, self.checkpoint_identifier_name, checkpoint_identifier_value)
        file_path = os.path.join(self.experiment_checkpoints_dir_path, file_name)

        # retrieve model parameters and random state
        experiment_checkpoint = self.get_experiment_checkpoint()

        successfully_saved = self.create_checkpoint_file(file_path, experiment_checkpoint)

        if successfully_saved and self.delete_old_checkpoints:
            self.delete_previous_checkpoint()

    def create_checkpoint_file(self, filepath: str, experiment_checkpoint: dict):
        """
        Creates a pickle file that contains the dictionary corresponding to the checkpoint
        :param filepath: path where the checkpoint is to be stored
        :param experiment_checkpoint: dictionary with data corresponding ot the current state of the experiment
        :return: bool, True if checkpoint was successfully saved
        """
        attempts = 10
        successfully_saved = False

        # attempt to save the experiment checkpoint
        for i in range(attempts):
            try:
                with open(filepath, mode="wb") as experiment_checkpoint_file:
                    pickle.dump(experiment_checkpoint, experiment_checkpoint_file)
                with open(filepath, mode="rb") as experiment_checkpoint_file:
                    pickle.load(experiment_checkpoint_file)
                successfully_saved = True
                break
            except ValueError:
                print("Something went wrong on attempt {0}.".format(i + 1))

        if successfully_saved:
            self._print("Checkpoint was successfully saved at:\n\t{0}".format(filepath))
        else:
            print("Something went wrong when attempting to save the experiment checkpoint.")

        return successfully_saved

    def delete_previous_checkpoint(self):
        """ Deletes the previous saved checkpoint """

        prev_ckpt_identifier_value = int(getattr(self, self.checkpoint_identifier_name) - self.checkpoint_save_frequency)
        file_name = "index-{0}_{1}-{2}.p".format(self.run_index, self.checkpoint_identifier_name, prev_ckpt_identifier_value)
        file_path = os.path.join(self.experiment_checkpoints_dir_path, file_name)

        if os.path.isfile(file_path):
            os.remove(file_path)
            print("The following file was deleted: {0}".format(file_path))

    def get_experiment_checkpoint(self):
        """ Creates a dictionary with all the necessary information to pause and resume the experiment """

        partial_results = {}
        for k, v in self.results_dict.items():
            partial_results[k] = v.cpu()

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

        return checkpoint

    def load_experiment_checkpoint(self):
        """
        Loads the latest experiment checkpoint
        """

        # find the file of the latest checkpoint
        file_name = self.get_latest_checkpoint_filename()
        if file_name == "":
            return False

        # get path to the latest checkpoint and check that it's a file
        file_path = os.path.join(self.experiment_checkpoints_dir_path, file_name)
        assert os.path.isfile(file_path)

        # load checkpoint information
        self._load_experiment_checkpoint(file_path)
        print("Experiment checkpoint successfully loaded from:\n\t{0}".format(file_path))
        return True

    def get_latest_checkpoint_filename(self):
        """
        gets the path to the file of the last saved checkpoint of the experiment
        """
        if not os.path.isdir(self.experiment_checkpoints_dir_path):
            return ""

        latest_checkpoint_id = 0
        latest_checkpoint_file_name = ""
        for file_name in os.listdir(self.experiment_checkpoints_dir_path):
            file_name_without_extension, _ = os.path.splitext(file_name)

            # Use regular expressions to find key-value pairs
            pairs = re.findall(r'(\w+)-(\d+)', file_name_without_extension)
            index_int = int(pairs[0][1])
            ckpt_id_int = int(pairs[1][1])

            if index_int != self.run_index:
                continue

            if ckpt_id_int > latest_checkpoint_id:
                latest_checkpoint_id = ckpt_id_int
                latest_checkpoint_file_name = file_name

        return latest_checkpoint_file_name

    def _load_experiment_checkpoint(self, file_path):
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
            self.results_dict[k] = partial_results[k].to(self.device)

    # ----------------------------- For storing summaries ----------------------------- #
    def _store_training_summaries(self):
        # store train data
        self.results_dict["train_loss_per_checkpoint"][self.current_running_avg_step] += self.running_loss / self.running_avg_window
        self.results_dict["train_accuracy_per_checkpoint"][self.current_running_avg_step] += self.running_accuracy / self.running_avg_window

        self._print("\t\tOnline accuracy: {0:.2f}".format(self.running_accuracy / self.running_avg_window))
        self.running_loss *= 0.0
        self.running_accuracy *= 0.0
        self.current_running_avg_step += 1

    def _store_test_summaries(self, test_data, epoch_number: int, epoch_runtime: float):
        """ Computes test summaries and stores them in results dir """
        self.results_dict["epoch_runtime"][epoch_number] += torch.tensor(epoch_runtime, dtype=torch.float32)
        # evaluate on test data
        self.net.eval()
        test_evaluation_start_time = time.perf_counter()
        test_loss, test_accuracy = self.evaluate_network(test_data)
        test_evaluation_end_time = time.perf_counter()
        self.net.train()
        # store parameters of the best accuracy model
        if test_accuracy > self.best_accuracy:
            self.best_accuracy = test_accuracy
            self.best_accuracy_model_parameters = deepcopy(self.net.state_dict())
        # store summaries
        evaluation_run_time = test_evaluation_end_time - test_evaluation_start_time
        self.results_dict["test_evaluation_runtime"][epoch_number] += torch.tensor(evaluation_run_time, dtype=torch.float32)
        self.results_dict["test_loss_per_epoch"][epoch_number] += test_loss
        self.results_dict["test_accuracy_per_epoch"][epoch_number] += test_accuracy
        # print progress
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

    # --------------------------- For running the experiment --------------------------- #
    def run(self):
        # load data
        training_data, training_dataloader = self.get_data(train=True, return_data_loader=True)
        test_data, test_dataloader = self.get_data(train=False, return_data_loader=True)

        self.load_experiment_checkpoint()
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
        cifar_type = 10 if not self.use_cifar100 else 100
        cifar_data = CifarDataSet(root_dir=self.data_path,
                                  train=train,
                                  cifar_type=100,
                                  device=None,
                                  image_normalization="max",
                                  label_preprocessing="one-hot",
                                  use_torch=True)
        mean = (0.5071, 0.4865, 0.4409) if self.use_cifar100 else (0.4914, 0.4822, 0.4465)
        std = (0.2673, 0.2564, 0.2762) if self.use_cifar100 else (0.2470, 0.2435, 0.2616)

        transformations = [
            ToTensor(swap_color_axis=True),  # reshape to (C x H x W)
            Normalize(mean=mean, std=std),  # center by mean and divide by std
        ]
        if train and self.use_data_augmentation:
            transformations.append(RandomHorizontalFlip(p=0.5))
            transformations.append(RandomCrop(size=32, padding=4, padding_mode="reflect"))
            transformations.append(RandomRotator(degrees=(0,15)))

        cifar_data.set_transformation(transforms.Compose(transformations))

        if return_data_loader:
            num_workers = 1 if self.device.type == "cpu" else 12
            dataloader = DataLoader(cifar_data, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
            return cifar_data, dataloader

        return cifar_data

    def train(self, train_dataloader: DataLoader, test_dataloader: DataLoader, test_data: CifarDataSet,
              training_data: CifarDataSet):

        training_data.select_new_partition(self.all_classes[:self.current_num_classes])
        test_data.select_new_partition(self.all_classes[:self.current_num_classes])

        for e in range(self.current_epoch, self.num_epochs):
            self._print("\tEpoch number: {0}".format(e + 1))
            self.set_lr()

            epoch_start_time = time.perf_counter()
            for step_number, sample in enumerate(train_dataloader):
                # sample observationa and target
                image = sample["image"].to(self.device)
                label = sample["label"].to(self.device)

                # reset gradients
                for param in self.net.parameters(): param.grad = None  # apparently faster than optim.zero_grad()

                # compute prediction and loss
                predictions = self.net.forward(image)[:, self.all_classes[:self.current_num_classes]]
                current_reg_loss = self.loss(predictions, label)
                current_loss = current_reg_loss.detach().clone()

                # backpropagate and update weights
                current_reg_loss.backward()
                self.optim.step()

                # store summaries
                current_accuracy = torch.mean((predictions.argmax(axis=1) == label.argmax(axis=1)).to(torch.float32))
                self.running_loss += current_loss
                self.running_accuracy += current_accuracy.detach()
                if (step_number + 1) % self.running_avg_window == 0:
                    self._print("\t\tStep Number: {0}".format(step_number + 1))
                    self._store_training_summaries()

            epoch_end_time = time.perf_counter()
            self._store_test_summaries(test_dataloader, epoch_number=e, epoch_runtime=epoch_end_time - epoch_start_time)

            self.current_epoch += 1
            self.extend_classes(training_data, test_data)

            if self.current_epoch % self.checkpoint_save_frequency == 0:
                self.save_experiment_checkpoint()

    def set_lr(self):
        """ Changes the learning rate of the optimizer according to the current epoch of the task """
        if not self.use_lr_schedule: return

        current_stepsize = None
        if (self.current_epoch % self.class_increase_frequency) == 0:
            current_stepsize = self.stepsize
        elif (self.current_epoch % self.class_increase_frequency) == 60:
            current_stepsize = round(self.stepsize * 0.2, 3)
        elif (self.current_epoch % self.class_increase_frequency) == 120:
            current_stepsize = round(self.stepsize * (0.2 ** 2), 3)
        elif (self.current_epoch % self.class_increase_frequency) == 160:
            current_stepsize = round(self.stepsize * (0.2 ** 3), 3)

        if current_stepsize is not None:
            for g in self.optim.param_groups:
                g['lr'] = current_stepsize
            self._print("\tCurrent stepsize: {0:.3f}".format(current_stepsize))

    def extend_classes(self, training_data: CifarDataSet, test_data: CifarDataSet):
        """
        Adds one new class to the data set with certain frequency
        """
        if self.current_num_classes == self.num_classes: return

        if (self.current_epoch % self.class_increase_frequency) == 0 and (not self.fixed_classes):
            self.net.load_state_dict(self.best_accuracy_model_parameters)
            self.best_accuracy = torch.zeros_like(self.best_accuracy)
            self.best_accuracy_model_parameters = {}
            self._save_model_parameters()
            increase = 1 if not self.use_cifar100 else 10
            self.current_num_classes += increase
            training_data.select_new_partition(self.all_classes[:self.current_num_classes])
            test_data.select_new_partition(self.all_classes[:self.current_num_classes])
            self._print("\tNew class added...")
            if self.reset_head:
                # kaiming_init_resnet_module(self.net.fc)                   # for resnet 10, 18 and 34
                kaiming_init_resnet_module(self.net.classifier[-1])  # for resnet 9
            if self.reset_network:
                self.net.apply(kaiming_init_resnet_module)

    def _save_model_parameters(self):
        """ Stores the parameters of the model, so it can be evaluated after the experiment is over """

        model_parameters_dir_path = os.path.join(self.results_dir, "model_parameters")
        os.makedirs(model_parameters_dir_path, exist_ok=True)

        file_name = "index-{0}_epoch-{1}.pt".format(self.run_index, self.current_epoch)
        file_path = os.path.join(model_parameters_dir_path, file_name)

        attempts = 10
        for i in range(attempts):
            try:
                torch.save(self.net.state_dict(), file_path)
                torch.load(file_path)                           # check that parameters were stored correctly
                break
            except ValueError:
                print("Something went wrong on attempt {0}! Attempting to store the parameters again...".format(i + 1))

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
        "stepsize": 0.1,
        "weight_decay": 0.0005,
        "momentum": 0.9,
        "data_path": os.path.join(file_path, "data"),
        "num_epochs": 200,
        "initial_num_classes": 100,
        "fixed_classes": True,
        "reset_head": False,
        "reset_network": False,
        "use_data_augmentation": True,
        "use_cifar100": True,
        "use_lr_schedule": True,
        "plot": False
    }

    print(experiment_parameters)
    relevant_parameters = ["num_epochs", "initial_num_classes", "fixed_classes", "stepsize", "weight_decay", "momentum",
                           "reset_head", "reset_network", "use_data_augmentation", "use_cifar100", "use_lr_schedule"]
    results_dir_name = "{0}-{1}".format(relevant_parameters[0], experiment_parameters[relevant_parameters[0]])
    for relevant_param in relevant_parameters[1:]:
        results_dir_name += "_" + relevant_param + "-" + str(experiment_parameters[relevant_param])

    initial_time = time.perf_counter()
    exp = IncrementalCIFARExperiment(experiment_parameters,
                                     results_dir=os.path.join(file_path, "results", results_dir_name),
                                     run_index=0,
                                     verbose=True)
    exp.run()
    exp.store_results()
    final_time = time.perf_counter()
    print("The running time in minutes is: {0:.2f}".format((final_time - initial_time) / 60))


if __name__ == "__main__":
    main()
