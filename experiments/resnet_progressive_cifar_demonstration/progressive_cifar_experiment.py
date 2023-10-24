# built-in libraries
import time
import os

# third party libraries
import torch
from torch.utils.data import DataLoader
import numpy as np
from torchvision.models import resnet18
from torchvision import transforms

# from ml project manager
from mlproj_manager.problems import CifarDataSet
from mlproj_manager.experiments import Experiment
from mlproj_manager.util import turn_off_debugging_processes, get_random_seeds, access_dict
from mlproj_manager.util.neural_networks import xavier_init_weights, get_activation_module
from mlproj_manager.util.data_preprocessing_and_transformations.image_transformations import ToTensor, Normalize, RandomHorizontalFlip, RandomRotator, RandomVerticalFlip


class ProgressiveCIFARExperiment(Experiment):

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
        self.weight_decay = exp_params["weight_decay"]
        self.momentum = exp_params["momentum"]
        self.gradient_clip_val = exp_params["gradient_clip_val"]
        self.data_path = exp_params["data_path"]
        self.num_epochs = access_dict(exp_params, "num_epochs", default=1, val_type=int)
        self.current_num_classes = access_dict(exp_params, "initial_num_classes", default=2, val_type=int)
        self.fixed_classes = access_dict(exp_params, "fixed_classes", default=True, val_type=bool)
        self.plot = access_dict(exp_params, key="plot", default=False)

        """ Training constants """
        self.batch_size = 100
        self.num_classes = 10
        self.image_dims = (32, 32, 3)
        self.flat_image_dims = int(np.prod(self.image_dims))
        self.num_images_per_epoch = 50000
        self.num_test_samples = 10000
        self.num_test_batches = self.num_test_samples / self.batch_size

        """ Network set up """
        # initialize network
        self.net = resnet18(num_classes=10, norm_layer=torch.nn.Identity)
        # self.net.apply(xavier_init_weights)

        # initialize optimizer
        self.optim = torch.optim.SGD(self.net.parameters(), lr=self.stepsize, momentum=self.momentum,
                                     weight_decay=self.weight_decay)

        # define loss function
        self.loss = torch.nn.CrossEntropyLoss(reduction="mean")

        # move network to device
        self.net.to(self.device)

        """ For summaries """
        self.checkpoint = 50
        self.current_ckpt, self.running_loss, self.running_accuracy = (0, 0.0, 0.0)
        self._initialize_summaries()

        """ For data partitioning """
        self.class_increase_frequency = 300
        self.all_classes = np.random.permutation(10)

    # -------------------- Methods for initializing the experiment --------------------#
    def _initialize_summaries(self):
        """
        Initializes the summaries for the experiment
        """
        total_checkpoints = self.num_images_per_epoch * self.num_epochs // (self.checkpoint * self.batch_size)
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

    # ----------------------------- For storing summaries ----------------------------- #
    def _store_training_summaries(self):
        # store train data
        self.results_dict["train_loss_per_checkpoint"][self.current_ckpt] += self.running_loss / self.checkpoint
        self.results_dict["train_accuracy_per_checkpoint"][self.current_ckpt] += self.running_accuracy / self.checkpoint

        self._print("\t\tOnline accuracy: {0:.2f}".format(self.running_accuracy / self.checkpoint))
        self.running_loss *= 0.0
        self.running_accuracy *= 0.0
        self.current_ckpt += 1

    def _store_test_summaries(self, test_data, epoch_number: int, epoch_runtime: float):
        """ Computes test summaries and stores them in results dir """
        self.results_dict["epoch_runtime"][epoch_number] += torch.tensor(epoch_runtime, dtype=torch.float32)
        # evaluate on test data
        self.net.eval()
        test_evaluation_start_time = time.perf_counter()
        test_loss, test_accuracy = self.evaluate_network(test_data)
        test_evaluation_end_time = time.perf_counter()
        self.net.train()
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
                test_predictions = self.net.forward(images.to(self.device))[:, self.all_classes[:self.current_num_classes]]

                avg_loss += self.loss(test_predictions, test_labels)
                avg_acc += torch.mean((test_predictions.argmax(axis=1) == test_labels.argmax(axis=1)).to(torch.float32))
                num_test_batches += 1

        return avg_loss / num_test_batches, avg_acc / num_test_batches

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
        cifar_data = CifarDataSet(root_dir=self.data_path,
                                  train=train,
                                  cifar_type=10,
                                  device=None,
                                  image_normalization="max",
                                  label_preprocessing="one-hot",
                                  use_torch=True)

        transformations = [
            ToTensor(swap_color_axis=True),  # reshape to (C x H x W)
            Normalize(mean=(0.491, 0.482, 0.446), std=(0.247, 0.243, 0.261)),  # center by mean and divide by std
        ]
        if train:
            transformations.append(RandomHorizontalFlip(p=0.5))
            transformations.append(RandomVerticalFlip(p=0.5))
            transformations.append(RandomRotator(degrees=(0,10)))
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

        for e in range(self.num_epochs):
            self._print("\tEpoch number: {0}".format(e + 1))

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
                torch.nn.utils.clip_grad_value_(self.net.parameters(), clip_value=self.gradient_clip_val)
                self.optim.step()

                # store summaries
                current_accuracy = torch.mean((predictions.argmax(axis=1) == label.argmax(axis=1)).to(torch.float32))
                self.running_loss += current_loss
                self.running_accuracy += current_accuracy.detach()
                if (step_number + 1) % self.checkpoint == 0:
                    self._print("\t\tStep Number: {0}".format(step_number + 1))
                    self._store_training_summaries()

            epoch_end_time = time.perf_counter()
            self._store_test_summaries(test_dataloader, epoch_number=e, epoch_runtime=epoch_end_time - epoch_start_time)

            if ((e + 1) % 300) == 0 and (not self.fixed_classes):
                self._print("\tNew class added...")
                self.current_num_classes += 1
                training_data.select_new_partition(self.all_classes[:self.current_num_classes])
                test_data.select_new_partition(self.all_classes[:self.current_num_classes])

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
        "stepsize": 0.01,
        "weight_decay": 0.0001,
        "momentum": 0.9,
        "gradient_clip_val": 0.1,
        "data_path": os.path.join(file_path, "data"),
        "num_epochs": 2700,
        "initial_num_classes": 2,
        "fixed_classes": False,
        "plot": False
    }

    print(experiment_parameters)
    relevant_parameters = ["num_epochs", "initial_num_classes", "fixed_classes", "stepsize", "weight_decay", "momentum",
                           "gradient_clip_val"]
    results_dir_name = "{0}-{1}".format(relevant_parameters[0], experiment_parameters[relevant_parameters[0]])
    for relevant_param in relevant_parameters[1:]:
        results_dir_name += "_" + relevant_param + "-" + str(experiment_parameters[relevant_param])

    initial_time = time.perf_counter()
    exp = ProgressiveCIFARExperiment(experiment_parameters,
                                     results_dir=os.path.join(file_path, "results", results_dir_name),
                                     run_index=0,
                                     verbose=True)
    exp.run()
    exp.store_results()
    final_time = time.perf_counter()
    print("The running time in minutes is: {0:.2f}".format((final_time - initial_time) / 60))


if __name__ == "__main__":
    main()
