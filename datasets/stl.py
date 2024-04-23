import copy
import math
from collections import Counter
import numpy as np
import random

from .basic_dataset import BasicDataset

import torchvision
from torchvision import transforms

def get_STL_transform(mean, std, train=True, size=32):
    """
    Get the data transformation for the STL dataset.

    Args:
        mean (list): Mean values for normalization.
        std (list): Standard deviation values for normalization.
        train (bool): If True, apply training transformations; otherwise, apply test transformations.
        size (int): Size of the dataset.

    Returns:
        torchvision.transforms.Compose: Composed transformations.
    """
    if train:
        return transforms.Compose([
            transforms.Resize(size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        return transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])


class STL_SSL_LT_Dataset:
    """
    SSL_Dataset class gets dataset (cifar10, cifar100) from torchvision.datasets,
    separates labeled and unlabeled data,
    and return BasicDataset: torch.utils.data.Dataset (see datasets.dataset.py)
    """
    def __init__(self, name='stl10', num_classes=10, data_dir='./data', N1=450,
                 include_train=False, imbalance_l=100.0, use_strong_transform=False, size=32):
        """
        Args:
            name (str): Name of the dataset in torchvision.datasets (stl10).
            num_classes (int): Number of label classes.
            data_dir (str): Path of the directory where data is downloaded or stored.
            N1 (int): Number of labeled samples.
            include_train (bool): Include labeled data in unlabeled data.
            imbalance_l (float): Imbalance factor for labeled data.
            use_strong_transform (bool): Use strong data augmentation.
            size (int): Size of the dataset.
        """
        self.name = name
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.N1 = N1
        self.include_train = include_train
        self.imbalance_l = imbalance_l

        self.mean = [x / 255 for x in [112.3, 108.9, 98.3]]
        self.std = [x / 255 for x in [68.5, 66.6, 68.4]]
        self.train_transform = get_STL_transform(self.mean, self.std, train=True, size=size)
        self.test_transform = get_STL_transform(self.mean, self.std, train=False, size=size)
        self.size = size
        self.use_strong_transform = use_strong_transform

        # get the labeled, unlabeled and training split for the data 
        # along with the classes information
        self.get_data()
        
        # apply long-tail sampling
        self.apply_longtail_sampling()
        print("ULB: ", Counter(self.targets_u))
        print("LB: ", Counter(self.targets_l))
        self.get_dataset()

    def get_data(self):
        """
        Get data (images) and targets (labels) from the STL10 dataset.
        """
        dset_l = torchvision.datasets.STL10("./data", split='train', download=True)
        dset_u = torchvision.datasets.STL10("./data", split='unlabeled', download=True)
        dset_test = torchvision.datasets.STL10("./data", split='test', download=True)
        self.data_l, self.targets_l, self.classes = dset_l.data, dset_l.labels, dset_l.classes
        self.data_u, self.targets_u = dset_u.data, dset_u.labels
        self.data_test, self.targets_test = dset_test.data, dset_test.labels
        return

    def apply_longtail_sampling(self):
        """
        Apply long-tail sampling to labeled and unlabeled data.

        This method simulates an imbalanced dataset by selecting samples from each class
        based on a long-tail distribution determined by the imbalance factor. It updates
        the labeled and validation data and targets accordingly.
        """
        dataset_class_wise = {}
        val_dataset_class_wise = {}

        for i in range(self.num_classes):
            dataset_class_wise[i] = []
            val_dataset_class_wise[i] = []

        for i in range(len(self.targets_l)):
            label = self.targets_l[i]
            dataset_class_wise[label].append(i)

        lamda = math.exp(-1 * math.log(self.imbalance_l) / (self.num_classes - 1))

        for i in range(self.num_classes):
            num_samples = int(lamda ** i * self.N1)
            random.shuffle(dataset_class_wise[i])
            dataset_class_wise[i] = dataset_class_wise[i][:num_samples]
            val_dataset_class_wise[i] = dataset_class_wise[i][-50:]

        select_list = [idx for sublist in dataset_class_wise.values() for idx in sublist]
        select_list_val = [idx for sublist in val_dataset_class_wise.values() for idx in sublist]

        self.data_l, self.data_val = self.data_l[np.array(select_list)], self.data_l[np.array(select_list_val)]
        self.targets_l, self.targets_val = [self.targets_l[x] for x in select_list], [self.targets_l[x] for x in select_list_val]
        return

    def get_dataset(self):
        """
        Create labeled, unlabeled, validation, and test datasets.

        Returns:
            BasicDataset: Labeled, unlabeled, validation, and test datasets.
        """
        self.lb_dset = BasicDataset(self.data_l, self.targets_l, self.classes, self.num_classes,
                                    self.train_transform, False, None)
        self.ulb_dset = BasicDataset(self.data_u, self.targets_u, self.classes, self.num_classes,
                                     self.train_transform, self.use_strong_transform, None)

        self.test_dset = BasicDataset(self.data_test, self.targets_test, self.classes, self.num_classes,
                                      self.test_transform, False, None)
        self.val_dset = BasicDataset(self.data_val, self.targets_val, self.classes, self.num_classes,
                                     self.test_transform, False, None)
        return

    def return_splits(self):
        """
        Return labeled, unlabeled, validation, and test datasets.

        Returns:
            tuple: Labeled, unlabeled, validation, and test datasets.
        """
        return self.lb_dset, self.ulb_dset, self.val_dset, self.test_dset
