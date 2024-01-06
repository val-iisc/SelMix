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
from .dataset import BasicDataset

import torchvision
from torchvision import datasets, transforms






class BasicDataset(Dataset):
    """
    BasicDataset returns a pair of image and labels (targets).
    If targets are not given, BasicDataset returns None as the label.
    This class supports strong augmentation for Fixmatch,
    and return both weakly and strongly augmented images.
    """
    def __init__(self,
                 data,
                 targets=None,
                 classes=None,
                 num_classes=None,
                 transform=None,
                 use_strong_transform=False,
                 strong_transform=None,
                 onehot=False,
                 *args, **kwargs):
        """
        Args
            data: x_data
            targets: y_data (if not exist, None)
            num_classes: number of label classes
            transform: basic transformation of data
            use_strong_transform: If True, this dataset returns both weakly and strongly augmented images.
            strong_transform: list of transformation functions for strong augmentation
            onehot: If True, label is converted into onehot vector.
        """
        super(BasicDataset, self).__init__()
        self.data = data
        self.targets = targets
        self.classes = classes
        self.num_classes = num_classes
        self.use_strong_transform = use_strong_transform
        
        self.transform = transform
        self.only_idx = False
        if use_strong_transform:
            if strong_transform is None:
                self.strong_transform = copy.deepcopy(transform)
                self.strong_transform.transforms.insert(0, RandAugment(3,5))
        else:
            self.strong_transform = strong_transform
        self.is_longtail = False
        self.total_samples = sum(Counter(self.targets).values())
        self.prior = [n/self.total_samples for n in Counter(self.targets).values()]


    def __getitem__(self, idx):
        """
        If strong augmentation is not used,
            return weak_augment_image, target
        else:
            return weak_augment_image, strong_augment_image, target
        """
        
        #set idx-th target
        if self.targets is None:
            target = None
        else:
            target = self.targets[idx]
            
            
        #set augmented images
            
        
        if self.transform is None:
            if self.only_idx:
                return idx, target
            else:
                img = self.data[idx]
                return transforms.ToTensor()(img), target
        else:
            img = self.data[idx]
            if isinstance(img, np.ndarray):
                try:
                    img = Image.fromarray(img)
                except:
                    img = img.transpose((1,2,0))
                    img = Image.fromarray(img)
            img_w = self.transform(img)
            if not self.use_strong_transform:
                return img_w, target
            else:
                return  img_w, self.strong_transform(img), target

    
    def __len__(self):
        return len(self.data)
    


def get_transform(mean, std, train=True, size=32):
    if train:
        return transforms.Compose([transforms.RandomHorizontalFlip(),
                                      transforms.RandomCrop(size, padding=4),
                                      transforms.ToTensor(), 
                                      transforms.Normalize(mean, std)])
    else:
        return transforms.Compose([transforms.ToTensor(), 
                                     transforms.Normalize(mean, std)])
                                
def get_STL_transform(mean, std, train=True, size=32):
    if train:
        return transforms.Compose([ transforms.Resize(size),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop(size, padding=4),
                                    transforms.ToTensor(), 
                                    transforms.Normalize(mean, std)])
    else:
        return transforms.Compose([ transforms.Resize(size),
                                    transforms.ToTensor(), 
                                    transforms.Normalize(mean, std)])
import random

class SSL_Dataset:
    """
    SSL_Dataset class gets dataset (cifar10, cifar100) from torchvision.datasets,
    separates labeled and unlabeled data,
    and return BasicDataset: torch.utils.data.Dataset (see datasets.dataset.py)
    """
    def __init__(self,
                 name='cifar100',
                 train=True,
                 num_classes=100,
                 data_dir='./data',
                 N1=1500,
                 M1=3000,
                 size=45000,
                 include_train=False,
                 uratio=2.0,
                 imbalance_l=100.0,
                 imbalance_u=100.0,
                 use_strong_transform=False,
                 inverted=False,
                 frac=0.0):
        """
        Args
            name: name of dataset in torchvision.datasets (cifar10, cifar100)
            train: True means the dataset is training dataset (default=True)
            num_classes: number of label classes
            data_dir: path of directory, wher data is downloaed or stored.
        """
        self.inverted = inverted
        self.name = name
        self.train = train
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.N1=N1
        self.M1=M1
        self.frac = frac
        self.size=size
        self.include_train = include_train
        self.uratio = uratio
        self.imbalance_l=imbalance_l
        self.imbalance_u=imbalance_u
        self.transform = get_transform(mean[name], std[name], train, 32 if 'cifar' in name else 96)
        self.use_strong_transform = use_strong_transform
        self.get_data()
        self.lb_data, self.lb_targets, self.ulb_data, self.ulb_targets = self.split()
        self.longtail()
        if self.include_train:
            print("Including lb data to ulb data")
            self.ulb_data = np.concatenate((self.lb_data, self.ulb_data), 0)
            self.ulb_targets = self.lb_targets + self.ulb_targets
        else:
            pass
        print("ULB : ",Counter(self.ulb_targets))
        print("LB : ", Counter(self.lb_targets))
        self.lb_dset, self.ulb_dset, self.val_dset = self.basic_dataset()
    def get_data(self):
        """
        get_data returns data (images) and targets (labels)
        """
        if 'cifar' in self.name:
            dset = getattr(torchvision.datasets, self.name.upper())
            dset = dset(self.data_dir, train=self.train, download=True)
            data, targets, classes = dset.data, dset.targets, dset.classes

            self.data, self.targets, self.val_data, self.val_targets = self.resize(data, targets)

            self.classes = classes
        elif 'STL' in self.name:
            dset_l = dataset = torchvision.datasets.STL10("./data", split='train', download=True)
            data, labels, classes = dset.data, dset.labels, dset.classes
            self.data, self.targets = data, labels

            self.classes = classes

        return 
    def drop_elements_randomly(self, input_list, p):
        if not (0 <= p <= 1):
            raise ValueError("Fraction 'p' must be in the range [0, 1].")

        # Calculate the number of elements to drop
        num_elements_to_drop = int(len(input_list) * p)

        # Randomly shuffle the list and drop the specified fraction of elements
        random.shuffle(input_list)
        result_list = input_list[num_elements_to_drop:]

        return result_list

    def resize(self, data, targets):
        if self.size==-1:
            return data, targets
        else:
            targets=np.array(targets)
            samples_per_class = self.size//self.num_classes
            data_, targets_, idx_ = [], [], []
            val_data, val_targets, val_idx  = [], [], []
            for c in range(self.num_classes):
                idx = np.where(targets == c)[0]
                idx = idx[:samples_per_class]
                idx_.extend(idx)
                data_.extend(data[idx])
                targets_.extend(targets[idx])
            val_idx = list(set(list(range(len(targets)))) -  set(idx_))
            val_targets = targets[val_idx].tolist()
            target_idx_dict = {}
            for i in range(self.num_classes):
                target_idx_dict[i] = []
            for idx, target in zip(val_idx, val_targets):
                target_idx_dict[target].append(idx)

            val_idx_ = []
            for i in range(self.num_classes):
                val_idx_ = val_idx_ + target_idx_dict[i][:int(self.frac * len(target_idx_dict[i]))]
            val_idx = val_idx_

            print("flipping val data horizontally")
            val_data, val_targets = data[val_idx], targets[val_idx]
            val_data = np.array([x for x in val_data] + [np.fliplr(x) for x in val_data])
            print("shape is :", val_data.shape)
            val_targets = np.array(val_targets.tolist() + val_targets.tolist())
            print("val targets", val_targets)
            return np.array(data_), targets_, val_data, val_targets

    def split(self):
        lb_data = []
        lbs = []
        lb_idx = []
        targets = np.array(self.targets)
        samples_per_class = int((len(self.targets)/(self.uratio + 1))//self.num_classes)
        for c in range(self.num_classes):
            idx = np.where(targets == c)[0]
            idx = np.random.choice(idx, samples_per_class, False)
            lb_idx.extend(idx)
            lb_data.extend(self.data[idx])
            lbs.extend(targets[idx])
        lb_data, lbs, lb_idx = np.array(lb_data), np.array(lbs), np.array(lb_idx)
        ulb_idx = np.array(sorted(list(set(range(len(self.data))) - set(lb_idx)))) #unlabeled_data index of data
        return lb_data, lbs, self.data[ulb_idx], targets[ulb_idx].tolist()

    def longtail(self):
        dataset_class_wise = {}

        for i in range(self.num_classes):
            dataset_class_wise[i] = []

        for i in range(len(self.lb_targets)):
            img, label = self.lb_data[i], self.lb_targets[i]
            dataset_class_wise[label].append(i)

        lamda = math.exp(-1 * math.log(self.imbalance_l)/(self.num_classes - 1))
        for i in range(self.num_classes):
            num_samples = max(int(lamda**i * self.N1), int(self.N1/self.imbalance_l))
            dataset_class_wise[i] = dataset_class_wise[i][:num_samples]

        select_list = []
        for i in range(self.num_classes):
            select_list = select_list + dataset_class_wise[i]
        
        self.lb_data = self.lb_data[np.array(select_list)]
        self.lb_targets = list(self.lb_targets[x] for x in select_list)

        
        
        dataset_class_wise = {}
        
        for i in range(self.num_classes):
            dataset_class_wise[i] = []

        for i in range(len(self.ulb_targets)):
            img, label = self.ulb_data[i], self.ulb_targets[i]
            dataset_class_wise[label].append(i)

        lamda = math.exp(-1 * math.log(self.imbalance_u)/(self.num_classes - 1))
        for i in range(self.num_classes):
            if self.imbalance_u > 1:
                num_samples = max(int(lamda**i * self.M1), int(self.M1/self.imbalance_u))
            else:
                num_samples = min(int(lamda**i * self.M1), int(self.M1/self.imbalance_u))
            dataset_class_wise[i] = dataset_class_wise[i][:num_samples]

        select_list = []
        for i in range(self.num_classes):
            select_list = select_list + dataset_class_wise[i]
        
        self.ulb_data = self.ulb_data[np.array(select_list)]
        self.ulb_targets = list(self.ulb_targets[x] for x in select_list)
        return 
     
    
    def basic_dataset(self):
        lb_dset = BasicDataset(self.lb_data, self.lb_targets, self.classes, self.num_classes, 
                               self.transform, False, None, False)

        ulb_dset = BasicDataset(self.ulb_data, self.ulb_targets, self.classes, self.num_classes, 
                               self.transform, self.use_strong_transform, None, False)

        val_dset = BasicDataset(self.val_data, self.val_targets, self.classes, self.num_classes, 
                               get_transform(mean[self.name], std[self.name], False), False, None, False)
        return lb_dset, ulb_dset, val_dset

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