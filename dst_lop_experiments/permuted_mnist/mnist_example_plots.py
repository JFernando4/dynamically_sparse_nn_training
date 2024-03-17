"""
Example plots of MNIST digits with different permutation apply to them
"""

# built-in libraries
import os
# third party libraries
import matplotlib.pyplot as plt
import numpy as np
# personal libraries
from mlproj_manager.problems import MnistDataSet
from mlproj_manager.util import Permute


def main():

    file_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(file_path, "data")
    plot_dir = os.path.join(file_path, "digit_plots")
    os.makedirs(plot_dir, exist_ok=True)

    mnist_data = MnistDataSet(root_dir=data_path, train=True,image_normalization="max")
    num_pixels = 784
    num_classes = 10
    num_permutations = 5

    digits = []

    digit_indices = []
    for c in range(num_classes):
        digit_indices.append(np.argwhere(mnist_data.data["labels"] == c).flatten()[0])

    for p in range(num_permutations):
        temp_digits = []
        for c in range(num_classes):
            temp_digits.append(mnist_data[digit_indices[c]]["image"])
        digits.append(temp_digits)
        mnist_data.set_transformation(Permute(np.random.permutation(num_pixels)))

    for p in range(num_permutations):
        for nc in range(num_classes):
            plt.imshow(digits[p][nc])
            plt.axis("off")
            plt.savefig(os.path.join(plot_dir, "digit-{0}_permutation-{1}.svg".format(nc, p)))
            plt.close()


if __name__ == "__main__":
    main()
