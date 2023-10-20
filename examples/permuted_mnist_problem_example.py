# built-in libraries
import time
import os

# third party libraries
import torch
from torch.utils.data import DataLoader
import numpy as np

# from ml project manager
from mlproj_manager.problems import MnistDataSet
from mlproj_manager.util import access_dict, Permute
from mlproj_manager.util.neural_networks import xavier_init_weights


class PermutedMNISTExperiment:

    def __init__(self, exp_params: dict, results_dir: str, run_index: int, plot=True):

        # define torch device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        """ For reproducibility """
        torch.random.manual_seed(run_index)
        np.random.seed(run_index)

        """ Experiment parameters """
        self.stepsize = exp_params["stepsize"]
        self.l1_factor = exp_params["l1_factor"]
        self.use_l1_regularization = (self.l1_factor > 0.0)
        self.l2_factor = exp_params["l2_factor"]
        self.use_l2_regularization = (self.l2_factor > 0.0)
        self.data_path = exp_params["data_path"]
        self.num_epochs = access_dict(exp_params, key="num_epochs", default=1, val_type=int)
        self.num_layers = access_dict(exp_params, key="num_layers", default=3, val_type=int)
        self.num_hidden = access_dict(exp_params, key="num_hidden", default=100, val_type=int)
        self.plot = plot
        self.results_dir = results_dir

        """ Training constants """
        self.batch_size = 1
        self.num_classes = 10
        self.num_inputs = 784
        self.num_images_per_epoch = 60000

        """ Network set up """
        self.net = torch.nn.Sequential()
        # hidden layers
        in_features = self.num_inputs
        for _ in range(self.num_layers):
            out_features = self.num_hidden
            self.net.append(torch.nn.Linear(in_features, out_features, bias=True))
            self.net.append(torch.nn.ReLU())
            in_features = out_features
        # output layer
        self.net.append(torch.nn.Linear(self.num_hidden, self.num_classes, bias=True))

        # initialize weights
        self.net.apply(xavier_init_weights)

        # initialize optimizer
        self.optim = torch.optim.SGD(self.net.parameters(), lr=self.stepsize)

        # define loss function
        self.loss = torch.nn.CrossEntropyLoss(reduction="mean")

        # move network to device
        self.net.to(self.device)

        """ Experiment Summaries """
        self.checkpoint = 100
        self.current_ckpt, self.running_loss, self.running_accuracy = (0, 0.0, 0.0)

        self.results_dict = {}
        total_checkpoints = self.num_images_per_epoch * self.num_epochs // (self.checkpoint * self.batch_size)
        self.results_dict["train_loss_per_checkpoint"] = torch.zeros(total_checkpoints, device=self.device,
                                                                     dtype=torch.float32)
        self.results_dict["train_accuracy_per_checkpoint"] = torch.zeros(total_checkpoints, device=self.device,
                                                                         dtype=torch.float32)

    # ----------------------------- For storing summaries ----------------------------- #
    def _store_training_summaries(self):

        # store train data
        self.results_dict["train_loss_per_checkpoint"][self.current_ckpt] += self.running_loss / self.checkpoint
        self.results_dict["train_accuracy_per_checkpoint"][self.current_ckpt] += self.running_accuracy / self.checkpoint

        print("\t\tOnline accuracy: {0:.2f}".format(self.running_accuracy / self.checkpoint))
        self.running_loss *= 0.0
        self.running_accuracy *= 0.0
        self.current_ckpt += 1

    # --------------------------- For running the experiment --------------------------- #
    def run(self):
        # load data
        mnist_train_data = MnistDataSet(root_dir=self.data_path, train=True, device=self.device,
                                        image_normalization="max", label_preprocessing="one-hot", use_torch=True)
        mnist_data_loader = DataLoader(mnist_train_data, batch_size=self.batch_size, shuffle=True)

        # train network
        self.train(mnist_data_loader=mnist_data_loader, training_data=mnist_train_data)

        # plot and stor results
        self._plot_results()
        self.store_results()

    def train(self, mnist_data_loader: DataLoader, training_data: MnistDataSet):

        for e in range(self.num_epochs):
            print("\tEpoch number: {0}".format(e + 1))

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
                if self.use_l1_regularization:
                    current_reg_loss += self.l1_factor * torch.sum(torch.hstack([torch.abs(p).sum() for p in self.net.parameters()]))
                if self.use_l2_regularization:
                    current_reg_loss += self.l2_factor * torch.sum(torch.hstack([torch.square(p).sum() for p in self.net.parameters()]))

                # backpropagate and update weights
                current_reg_loss.backward()
                self.optim.step()

                # store summaries
                current_accuracy = torch.mean((predictions.argmax(axis=1) == label.argmax(axis=1)).to(torch.float32))
                self.running_loss += current_loss
                self.running_accuracy += current_accuracy.detach()
                if (i + 1) % self.checkpoint == 0:
                    print("\t\tStep Number: {0}".format(i + 1))
                    self._store_training_summaries()

            training_data.set_transformation(Permute(np.random.permutation(self.num_inputs)))

    def _plot_results(self):
        if self.plot:
            import matplotlib.pyplot as plt

            for rname, rvals in self.results_dict.items():
                plt.plot(torch.arange(rvals.size()[0]), rvals)
                plt.title(rname)
                plt.show()
                plt.close()

    def store_results(self):
        os.makedirs(self.results_dir,exist_ok=True)
        for k, v in self.results_dict.items():
            file_name = "{0}.npy".format(k)
            file_path = os.path.join(self.results_dir, file_name)
            np.save(file_path, v.numpy())


def main():
    """
    This is a quick demonstration of how to run the experiments. For a more systematic run, use the mlproj_manager
    scheduler.
    """
    file_path = os.path.dirname(os.path.abspath(__file__))
    experiment_parameters = {
        "stepsize": 0.001,
        "l1_factor": 0.0,
        "l2_factor": 0.0,
        "data_path": os.path.join(file_path, "data"),
        "num_epochs": 20,
        "num_layers": 3,
        "num_hidden": 100,
        "permute_inputs": False,
        "plot": True
    }

    print(experiment_parameters)
    initial_time = time.perf_counter()
    exp = PermutedMNISTExperiment(experiment_parameters,
                                  results_dir=os.path.join(file_path, "results"),
                                  run_index=0,
                                  plot=True)
    exp.run()
    exp.store_results()
    final_time = time.perf_counter()
    print("The running time in minutes is: {0:.2f}".format((final_time - initial_time) / 60))


if __name__ == "__main__":
    main()
