# third party libraries
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
# mlproj manager
from mlproj_manager.problems import CifarDataSet
from mlproj_manager.util.data_preprocessing_and_transformations import ToTensor, Normalize, RandomCrop, \
    RandomHorizontalFlip, RandomRotator


def get_cifar_data(data_path: str, train: bool = True, validation: bool = False, use_cifar100: bool = True,
                   use_data_augmentation: bool = True, batch_size: int = 90, num_workers: int = 12):
    """
    Loads the CIFAR-10 or 100 data set
    :param data_path: path to where the data is stored in memroy
    :param train: (bool) indicates whether to load the train (True) or the test (False) data
    :param validation: (bool) indicates whether to return the validation set. The validation set is made up of
                       50 examples of each class of whichever set was loaded
    :param use_cifar100: (bool) indicating whether to use CIFAR-100 or 10
    :param use_data_augmentation: (bool) indicating whether to use data augmentation
    :param batch_size: (int) batch size for the data loader
    :param num_workers: (int) number of workers for the data loader
    :return: data set, data loader
    """

    """ Loads CIFAR data set """
    cifar_type = 100 if use_cifar100 else 10
    cifar_data = CifarDataSet(root_dir=data_path,
                              train=train,
                              cifar_type=cifar_type,
                              device=None,
                              image_normalization="max",
                              label_preprocessing="one-hot",
                              use_torch=True)

    mean = (0.5071, 0.4865, 0.4409) if use_cifar100 else (0.4914, 0.4822, 0.4465)
    std = (0.2673, 0.2564, 0.2762) if use_cifar100 else (0.2470, 0.2435, 0.2616)

    transformations = [
        ToTensor(swap_color_axis=True),  # reshape to (C x H x W)
        Normalize(mean=mean, std=std),  # center by mean and divide by std
    ]

    if train and use_data_augmentation and (not validation):
        transformations.append(RandomHorizontalFlip(p=0.5))
        transformations.append(RandomCrop(size=32, padding=4, padding_mode="reflect"))
        transformations.append(RandomRotator(degrees=(0, 15)))

    cifar_data.set_transformation(transforms.Compose(transformations))

    if not train:
        dataloader = DataLoader(cifar_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        return cifar_data, dataloader

    train_indices, validation_indices = get_validation_and_train_indices(cifar_data)
    indices = validation_indices if validation else train_indices
    subsample_cifar_data_set(sub_sample_indices=indices, cifar_data=cifar_data)
    return cifar_data, DataLoader(cifar_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)


def get_validation_and_train_indices(cifar_data: CifarDataSet):
    """
    Splits the cifar data into validation and train set and returns the indices of each set with respect to the
    original dataset
    :param cifar_data: and instance of CifarDataSet
    :return: train and validation indices
    """
    num_classes = 100
    num_val_samples_per_class = 50
    num_train_samples_per_class = 450
    validation_set_size = 5000
    train_set_size = 45000

    validation_indices = torch.zeros(validation_set_size, dtype=torch.int32)
    train_indices = torch.zeros(train_set_size, dtype=torch.int32)
    current_val_samples = 0
    current_train_samples = 0
    for i in range(num_classes):
        class_indices = torch.argwhere(cifar_data.data["labels"][:, i] == 1).flatten()
        validation_indices[current_val_samples:(current_val_samples + num_val_samples_per_class)] += class_indices[:num_val_samples_per_class]
        train_indices[current_train_samples:(current_train_samples + num_train_samples_per_class)] += class_indices[num_val_samples_per_class:]
        current_val_samples += num_val_samples_per_class
        current_train_samples += num_train_samples_per_class

    return train_indices, validation_indices


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
