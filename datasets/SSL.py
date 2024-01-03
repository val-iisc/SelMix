import numpy as np
import copy 


from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler

from .augmentation.randaugment import RandAugment

import torchvision
from torchvision import datasets, transforms

class CIFAR10:
    '''
    SSL LT dataset for CIFAR10
    '''
    def __init__(self, 
                 r=2,
                 include_train=False,
                 imbalance_l=100,
                 imbalance_u=100,
                 N1=1500,
                 M1=3000,
                 data_dir='./data',
                 strong_l=False,
                 strong_u=True):
        self.r = r
        self.include_train = include_train
        self.imbalance_l = imbalance_l
        self.imbalance_u = imbalance_u
        if imbalance_u != imbalance_l:
            print("Mismatched distributions")
        else:
            print("Matched distributions")
        self.N1 = N1
        self.M1= M1
        self.data_dir = data_dir
        self.strong_l = strong_l
        self.strong_u = strong_u
        self.num_classes = 10

        self.mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        self.std = [x / 255 for x in [63.0, 62.1, 66.7]]
        self.cifar10_size = 50000
        self.num_val_samples = 5000

        self.aug_w = transforms.Compose([transforms.RandomHorizontalFlip(),
                                      transforms.RandomCrop(32, padding=4),
                                      transforms.ToTensor(), 
                                      transforms.Normalize(self.mean, self.std)])
        self.aug_s = copy.deepcopy(self.aug_w)
        self.aug_s.transforms.insert(0, RandAugment(3,5))
        self.eval_transform = transforms.Compose([transforms.ToTensor(), 
                                     transforms.Normalize(self.mean, self.std)])

        self.dataset_test =  CIFAR10(root=data_dir, train=False, transforms=self.eval_transform)
        self.lb, self.ulb, self.val = self.get_train_dataset()
        self.lb, self.ulb = self.longtail()

    def get_train_dataset(self):
        train = torchvision.datasets.CIFAR10(root=self.data_dir, train=True, transforms=self.aug_w)
        val = torchvision.datasets.CIFAR10(root=self.data_dir, train=True, transforms=self.aug_w)
        
        train_idx, val_idx = [], []
        lb_idx, ulb_idx= [], []
        num_train_samples = self.cifar10_size - self.num_val_samples
        num_lb_samples  = num_train_samples//(self.r + 1)
        num_ulb_samples = (num_train_samples * self.r)//(self.r + 1)

        targets = train.targets
        train_samples_per_class = num_train_samples // self.num_classes
        for c in range(self.num_classes):
            idx = np.where(targets == c)[0]
            idx = idx[:train_samples_per_class]
            train_idx.extend(idx)
        
        val_idx = list(set(list(range(len(targets)))) -  set(train_idx))
        
        val.data = val.data[np.array(val_idx)]
        val.targets = np.array(val.targets)[np.array(val_idx)].tolist()

        train.data = train.data[np.array(train_idx)]
        train.targets = np.array(train.targets)[np.array(train_idx)].tolist()

        lb_samples_per_class = num_lb_samples //self.num_classes
        for c in range(self.num_classes):
            idx = np.where(train.targets == c)[0]
            idx = np.random.choice(idx, lb_samples_per_class, False)
            lb_idx.extend(idx)
        
        ulb_idx = list(set(list(range(len(train.targets)))) -  set(lb_idx))

        lb = copy.deepcopy(train)
        ulb = copy.deepcopy(train)

        lb.data = lb.data[np.array(lb_idx)]
        lb.targets = np.array(lb.targets)[np.array(lb_idx)].tolist()

        ulb.data = ulb.data[np.array(ulb_idx)]
        ulb.targets = np.array(ulb.targets)[np.array(ulb_idx)].tolist()
        
        