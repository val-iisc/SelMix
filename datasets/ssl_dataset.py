import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler

from .augmentation.randaugment import RandAugment
from .data_utils import get_sampler_by_name, get_data_loader, get_onehot, split_ssl_data
from .basic_dataset import BasicDataset

import torchvision
from torchvision import datasets, transforms

mean, std = {}, {}
mean['cifar10'] = [x / 255 for x in [125.3, 123.0, 113.9]]
mean['cifar100'] = [x / 255 for x in [129.3, 124.1, 112.4]]

std['cifar10'] = [x / 255 for x in [63.0, 62.1, 66.7]]
std['cifar100'] = [x / 255 for x in [68.2,  65.4,  70.4]]


def get_transform(mean, std, train=True):
    if train:
        return transforms.Compose([transforms.RandomHorizontalFlip(),
                                      transforms.RandomCrop(32, padding=4),
                                      transforms.ToTensor(), 
                                      transforms.Normalize(mean, std)])
    else:
        return transforms.Compose([transforms.ToTensor(), 
                                     transforms.Normalize(mean, std)])


class SSL_Dataset:
    """
    SSL_Dataset class gets dataset (cifar10, cifar100) from torchvision.datasets,
    separates labeled and unlabeled data,
    and return BasicDataset: torch.utils.data.Dataset (see datasets.dataset.py)
    """
    def __init__(self,
                 name='cifar10',
                 train=True,
                 num_classes=10,
                 data_dir='./data'):
        """
        Args
            name: name of dataset in torchvision.datasets (cifar10, cifar100)
            train: True means the dataset is training dataset (default=True)
            num_classes: number of label classes
            data_dir: path of directory, where data is downloaed or stored.
        """
        
        self.name = name
        self.train = train
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.transform = get_transform(mean[name], std[name], train)
        
    def get_data(self):
        """
        get_data returns data (images) and targets (labels)
        """
        dset = getattr(torchvision.datasets, self.name.upper())
        dset = dset(self.data_dir, train=self.train, download=True)
        data, targets, classes = dset.data, dset.targets, dset.classes
        return data, targets, classes


    def get_dset(self, use_strong_transform=False, 
                 strong_transform=None, onehot=False):
        """
        get_dset returns class BasicDataset, containing the returns of get_data.
        
        Args
            use_strong_tranform: If True, returned dataset generates a pair of weak and strong augmented images.
            strong_transform: list of strong_transform (augmentation) if use_strong_transform is True
            onehot: If True, the label is not integer, but one-hot vector.
        """
        
        data, targets, classes = self.get_data()
        num_classes = self.num_classes
        transform = self.transform
        data_dir = self.data_dir
        
        return BasicDataset(data, targets, classes, num_classes, transform, 
                            use_strong_transform, strong_transform, onehot)


    def get_ssl_dset(self, index=None, include_lb_to_ulb=True,
                            use_strong_transform=False, strong_transform=None, 
                            onehot=False):
        """
        get_ssl_dset split training samples into labeled and unlabeled samples.
        The labeled data is balanced samples over classes.
        
        Args:
            num_labels: number of labeled data.
            index: If index of np.array is given, labeled data is not randomly sampled, but use index for sampling.
            include_lb_to_ulb: If True, consistency regularization is also computed for the labeled data.
            use_strong_transform: If True, unlabeld dataset returns weak & strong augmented image pair. 
                                  If False, unlabeled datasets returns only weak augmented image.
            strong_transform: list of strong transform (RandAugment in FixMatch)
            oenhot: If True, the target is converted into onehot vector.
            
        Returns:
            BasicDataset (for labeled data), BasicDataset (for unlabeld data)
        """
        
        data, targets, classes = self.get_data()
        num_classes = self.num_classes
        transform = self.transform
        data_dir = self.data_dir

        
        dset = BasicDataset(data, targets, classes, num_classes, 
                               transform, False, None, onehot)
        
        return  dset
