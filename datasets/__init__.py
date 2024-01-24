import numpy as np
import math
from collections import Counter
import copy 
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler

from .augmentation.randaugment import RandAugment
from .data_utils import get_sampler_by_name, get_data_loader, get_onehot, split_ssl_data
from .basic_dataset import BasicDataset

import torchvision
from torchvision import datasets, transforms


from .cifar import CIFAR_SSL_LT_Dataset
from .stl import STL_SSL_LT_Dataset



class SSL_STL_Dataset:
    """
    SSL_Dataset class gets dataset (cifar10, cifar100) from torchvision.datasets,
    separates labeled and unlabeled data,
    and return BasicDataset: torch.utils.data.Dataset (see datasets.dataset.py)
    """
    def __init__(self,
                 name='stl10',
                 num_classes=10,
                 data_dir='./data',
                 N1=500,
                 include_train=False,
                 imbalance_l=100.0,
                 use_strong_transform=False,
                 size=32):
        """
        Args
            name: name of dataset in torchvision.datasets (cifar10, cifar100)
            train: True means the dataset is training dataset (default=True)
            num_classes: number of label classes
            data_dir: path of directory, wher data is downloaed or stored.
        """
        self.name = name
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.N1=N1
        self.include_train = include_train
        self.imbalance_l=imbalance_l
        self.transform = get_STL_transform(mean[name], std[name], train=True, size=size)
        self.size = size
        self.use_strong_transform = use_strong_transform
        self.get_data()
        self.longtail()
        print("ULB : ",Counter(self.targets_u))
        print("LB : ", Counter(self.targets_l))

    def get_data(self):
        """
        get_data returns data (images) and targets (labels)
        """
        dset_l = torchvision.datasets.STL10("./data", split='train', download=True)
        dset_u = torchvision.datasets.STL10("./data", split='unlabeled', download=True)
        dset_test = torchvision.datasets.STL10("./data", split='test', download=True)
        self.data_l, self.targets_l, self.classes = dset_l.data, dset_l.labels, dset_l.classes
        self.data_u, self.targets_u = dset_u.data, dset_u.labels
        self.data_test, self.targets_test = dset_test.data, dset_test.labels
        return 


    def longtail(self):
        dataset_class_wise = {}
        val_dataset_class_wise = {}

        for i in range(self.num_classes):
            dataset_class_wise[i] = []

        for i in range(len(self.targets_l)):
            img, label = self.data_l[i], self.targets_l[i]
            dataset_class_wise[label].append(i)

        val_dataset_class_wise = copy.deepcopy(dataset_class_wise)

        lamda = math.exp(-1 * math.log(self.imbalance_l)/(self.num_classes - 1))
        for i in range(self.num_classes):
            num_samples = int(lamda**i * self.N1)
            dataset_class_wise[i] = dataset_class_wise[i][:num_samples]
            val_dataset_class_wise[i] = val_dataset_class_wise[i][-50:]

        select_list = []
        select_list_val = []
        for i in range(self.num_classes):
            select_list = select_list + dataset_class_wise[i]
            select_list_val = select_list_val + val_dataset_class_wise[i]

        data_l, targets_l = self.data_l, self.targets_l
        self.data_l, self.data_val = data_l[np.array(select_list)], data_l[np.array(select_list_val)]
        self.targets_l, self.targets_val = list(targets_l[x] for x in select_list), list(targets_l[x] for x in select_list_val)
        return 
     
    def basic_dataset(self):
        lb_dset = BasicDataset(self.data_l, self.targets_l, self.classes, self.num_classes, 
                               self.transform, True, None, False)
        ulb_dset = BasicDataset(self.data_u, self.targets_u, self.classes, self.num_classes, 
                               self.transform, self.use_strong_transform, None, False)

        test_transform = get_STL_transform(mean['stl10'], std['stl10'], train=False, size=self.size)
        test_dset = BasicDataset(self.data_test, self.targets_test, self.classes, self.num_classes, 
                                 test_transform, False, None, False)
        val_dset = BasicDataset(self.data_val, self.targets_val, self.classes, self.num_classes, 
                                 test_transform, False, None, False)
        return lb_dset, ulb_dset, test_dset, val_dset