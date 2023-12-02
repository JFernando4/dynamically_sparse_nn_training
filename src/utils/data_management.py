# third party libraries
import torch
# mlproj manager
from mlproj_manager.problems import CifarDataSet


def subsample_cifar_data_set(sub_sample_indices, cifar_data: CifarDataSet):
    """
    Sub-samples the CIFAR 100 data set according to the given indices
    :param sub_sample_indices: array of indices in the same format as the cifar data set (numpy or torch)
    :param cifar_data: cifar data to be sub-sampled
    :return: None, but modifies the given cifar_dataset
    """

    cifar_data.data["data"] = cifar_data.data["data"][sub_sample_indices]
    cifar_data.data["labels"] = cifar_data.data["labels"][sub_sample_indices]
    cifar_data.integer_labels = torch.tensor(cifar_data.integer_labels)[sub_sample_indices].tolist()
    cifar_data.current_data = cifar_data.partition_data()
