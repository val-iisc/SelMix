import numpy as np
import math
import random
from collections import Counter
from .basic_dataset import BasicDataset
import torchvision
from torchvision import transforms



def get_transform(mean, std, train=True, size=32):
    if train:
        return transforms.Compose([transforms.RandomHorizontalFlip(),
                                      transforms.RandomCrop(size, padding=4),
                                      transforms.ToTensor(), 
                                      transforms.Normalize(mean, std)])
    else:
        return transforms.Compose([transforms.ToTensor(), 
                                     transforms.Normalize(mean, std)])

class SSL_LT_Dataset:
    """
    SSL_Dataset class for handling labeled and unlabeled datasets.
    """
    def __init__(self, name='cifar100', num_classes=100, data_dir='./data',
                 N1=150, M1=300, train_set_size=45000, include_train=False, uratio=2.0,
                 imbalance_l=10.0, imbalance_u=10.0, use_strong_transform=False,inverted=False):
        """
        Initialize SSL_Dataset.

        Args:
            name: Name of the dataset in torchvision.datasets (cifar10, cifar100)
            train: True for the training dataset (default=True)
            num_classes: Number of label classes
            data_dir: Path of the directory where data is downloaded or stored.
            N1: Number of labeled samples
            M1: Number of unlabeled samples
            size: Size of the dataset
            include_train: Include labeled data to unlabeled data
            uratio: Ratio of unlabeled to labeled data
            imbalance_l: Imbalance factor for labeled data
            imbalance_u: Imbalance factor for unlabeled data
            use_strong_transform: Use strong data augmentation
            inverted: Invert dataset
        """
        self.inverted = inverted
        self.name = name
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.N1 = N1
        self.M1 = M1
        self.train_set_size = train_set_size
        self.include_train = include_train
        self.uratio = uratio
        self.imbalance_l = imbalance_l
        self.imbalance_u = imbalance_u
        self.mean, self.std = {}, {}

        self.mean['cifar10'] = [x / 255 for x in [125.3, 123.0, 113.9]]
        self.mean['cifar100'] = [x / 255 for x in [129.3, 124.1, 112.4]]

        self.std['cifar10'] = [x / 255 for x in [63.0, 62.1, 66.7]]
        self.std['cifar100'] = [x / 255 for x in [68.2,  65.4,  70.4]]

        self.train_transform = get_transform(self.mean[name], self.std[name], train=True, size=32)
        self.test_transform = get_transform(self.mean[name], self.std[name], train=False, size=32)

        self.use_strong_transform = use_strong_transform
        self.get_data()
        self.lb_data, self.lb_targets, self.ulb_data, self.ulb_targets = self.split()
        self.longtail()

        if self.include_train:
            print("Including labeled data to unlabeled data")
            self.ulb_data = np.concatenate((self.lb_data, self.ulb_data), 0)
            self.ulb_targets = self.lb_targets + self.ulb_targets
        else:
            pass

        print("Unlabeled: ", Counter(self.ulb_targets))
        print("Labeled: ", Counter(self.lb_targets))

        self.lb_dset, self.ulb_dset, self.val_dset = self.get_dataset()

    def get_data(self):
        """
        Get data (images), targets (labels), and classes.
        """
        dset = getattr(torchvision.datasets, self.name.upper())
        dset = dset(self.data_dir, train=True, download=True)
        data, targets, classes = dset.data, dset.targets, dset.classes
        self.data, self.targets, self.val_data, self.val_targets = self.resize(data, targets) # type: ignore
        self.classes = classes
        self.test_dset = dset(self.data_dir, train=False, transforms=self.test_transform, download=True)

    def drop_elements_randomly(self, input_list, p):
        """
        Drop elements randomly from the input list.

        Args:
            input_list: List from which elements are dropped.
            p: Fraction of elements to drop.

        Returns:
            Resulting list after dropping elements.
        """
        if not (0 <= p <= 1):
            raise ValueError("Fraction 'p' must be in the range [0, 1].")

        # Calculate the number of elements to drop
        num_elements_to_drop = int(len(input_list) * p)

        # Randomly shuffle the list and drop the specified fraction of elements
        random.shuffle(input_list)
        result_list = input_list[num_elements_to_drop:]

        return result_list

    def resize(self, data, targets):
        """
        Resize the dataset.

        Args:
            data: Input data.
            targets: Target labels.

        Returns:
            Resized data and targets.
        """
        targets = np.array(targets)
        samples_per_class = self.train_set_size // self.num_classes
        data_, targets_, idx_ = [], [], []
        val_data, val_targets, val_idx = [], [], []

        for c in range(self.num_classes):
            idx = np.where(targets == c)[0]
            idx = idx[:samples_per_class]
            idx_.extend(idx)
            data_.extend(data[idx])
            targets_.extend(targets[idx])

        val_idx = list(set(list(range(len(targets)))) - set(idx_))
        val_targets = targets[val_idx].tolist()

        target_idx_dict = {}
        for i in range(self.num_classes):
            target_idx_dict[i] = []

        for idx, target in zip(val_idx, val_targets):
            target_idx_dict[target].append(idx)

        val_idx_ = []
        for i in range(self.num_classes):
            val_idx_ = val_idx_ + target_idx_dict[i]
        val_idx = val_idx_

        val_data, val_targets = data[val_idx], targets[val_idx]
        val_data = np.array([x for x in val_data] + [np.fliplr(x) for x in val_data])
        val_targets = np.array(val_targets.tolist() + val_targets.tolist())

        return np.array(data_), targets_, val_data, val_targets

    def split(self):
        """
        Split the dataset into labeled and unlabeled samples.

        Returns:
            Labeled data, labeled targets, unlabeled data, and unlabeled targets.
        """
        lb_data = []
        lbs = []
        lb_idx = []
        targets = np.array(self.targets)
        samples_per_class = int((len(self.targets) / (self.uratio + 1)) // self.num_classes) # type: ignore

        for c in range(self.num_classes):
            idx = np.where(targets == c)[0]
            idx = np.random.choice(idx, samples_per_class, False)
            lb_idx.extend(idx)
            lb_data.extend(self.data[idx])
            lbs.extend(targets[idx])

        lb_data, lbs, lb_idx = np.array(lb_data), np.array(lbs), np.array(lb_idx)
        ulb_idx = np.array(sorted(list(set(range(len(self.data))) - set(lb_idx))))
        return lb_data, lbs, self.data[ulb_idx], targets[ulb_idx].tolist()

    def longtail(self):
        """
        Apply long-tail sampling to labeled and unlabeled data.
        """
        dataset_class_wise = {}

        for i in range(self.num_classes):
            dataset_class_wise[i] = []

        for i in range(len(self.lb_targets)):
            img, label = self.lb_data[i], self.lb_targets[i]
            dataset_class_wise[label].append(i)

        lamda = math.exp(-1 * math.log(self.imbalance_l) / (self.num_classes - 1))
        for i in range(self.num_classes):
            num_samples = max(int(lamda ** i * self.N1), int(self.N1 / self.imbalance_l))
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

        lamda = math.exp(-1 * math.log(self.imbalance_u) / (self.num_classes - 1))
        for i in range(self.num_classes):
            if self.imbalance_u > 1:
                num_samples = max(int(lamda ** i * self.M1), int(self.M1 / self.imbalance_u))
            else:
                num_samples = min(int(lamda ** i * self.M1), int(self.M1 / self.imbalance_u))
            dataset_class_wise[i] = dataset_class_wise[i][:num_samples]

        select_list = []
        for i in range(self.num_classes):
            select_list = select_list + dataset_class_wise[i]

        self.ulb_data = self.ulb_data[np.array(select_list)]
        self.ulb_targets = list(self.ulb_targets[x] for x in select_list)

    def get_dataset(self):
        """
        Create labeled, unlabeled, and validation datasets.

        Returns:
            Labeled dataset, unlabeled dataset, and validation dataset.
        """
        lb_dset = BasicDataset(self.lb_data, self.lb_targets, self.classes, self.num_classes,
                               self.train_transform, False, None, False)

        ulb_dset = BasicDataset(self.ulb_data, self.ulb_targets, self.classes, self.num_classes,
                                self.train_transform, self.use_strong_transform, None, False)

        val_dset = BasicDataset(self.val_data, self.val_targets, self.classes, self.num_classes,
                                self.test_transform, False, None, False)
        return lb_dset, ulb_dset, val_dset


class LT_Dataset:
    """
    SSL_Dataset class for handling labeled and unlabeled datasets.
    """
    def __init__(self, name='cifar100', num_classes=100,
                 data_dir='./data', N1=150, train_set_size=45000,
                 imbalance_l=10.0, use_strong_transform=False):
        """
        Initialize SSL_Dataset.

        Args:
            name: Name of the dataset in torchvision.datasets (cifar10, cifar100)
            train: True for the training dataset (default=True)
            num_classes: Number of label classes
            data_dir: Path of the directory where data is downloaded or stored.
            N1: Number of labeled samples
            size: Size of the dataset
            uratio: Ratio of unlabeled to labeled data
            imbalance_l: Imbalance factor for labeled data
            use_strong_transform: Use strong data augmentation
            inverted: Invert dataset
        """
        self.name = name
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.N1 = N1
        self.train_set_size = train_set_size
        self.imbalance_l = imbalance_l
        self.mean, self.std = {}, {}

        self.mean['cifar10'] = [x / 255 for x in [125.3, 123.0, 113.9]]
        self.mean['cifar100'] = [x / 255 for x in [129.3, 124.1, 112.4]]

        self.std['cifar10'] = [x / 255 for x in [63.0, 62.1, 66.7]]
        self.std['cifar100'] = [x / 255 for x in [68.2,  65.4,  70.4]]

        self.train_transform = get_transform(self.mean[name], self.std[name], True, 32)
        self.test_transform = get_transform(self.mean[name], self.std[name], False, 32)
        self.use_strong_transform = use_strong_transform
        self.get_data()
        self.longtail()
        
        print("Labeled: ", Counter(self.lb_targets))
        print("Val: ", Counter(self.val_targets))

        self.lb_dset,  self.val_dset = self.get_dataset()

    def get_data(self):
        """
        Get data (images), targets (labels), and classes.
        """
        dset = getattr(torchvision.datasets, self.name.upper())
        dset = dset(self.data_dir, train=True, download=True)
        data, targets, classes = dset.data, dset.targets, dset.classes
        self.lb_data, self.lb_targets, self.val_data, self.val_targets = self.resize(data, targets) # type: ignore
        self.classes = classes
        self.test_dset = dset(self.data_dir, train=True, download=True, transforms=self.test_transform)

    def drop_elements_randomly(self, input_list, p):
        """
        Drop elements randomly from the input list.

        Args:
            input_list: List from which elements are dropped.
            p: Fraction of elements to drop.

        Returns:
            Resulting list after dropping elements.
        """
        if not (0 <= p <= 1):
            raise ValueError("Fraction 'p' must be in the range [0, 1].")

        # Calculate the number of elements to drop
        num_elements_to_drop = int(len(input_list) * p)

        # Randomly shuffle the list and drop the specified fraction of elements
        random.shuffle(input_list)
        result_list = input_list[num_elements_to_drop:]

        return result_list

    def resize(self, data, targets):
        """
        Resize the dataset.

        Args:
            data: Input data.
            targets: Target labels.

        Returns:
            Resized data and targets.
        """
        targets = np.array(targets)
        samples_per_class = self.train_set_size // self.num_classes
        data_, targets_, idx_ = [], [], []
        val_data, val_targets, val_idx = [], [], []

        for c in range(self.num_classes):
            idx = np.where(targets == c)[0]
            idx = idx[:samples_per_class]
            idx_.extend(idx)
            data_.extend(data[idx])
            targets_.extend(targets[idx])

        val_idx = list(set(list(range(len(targets)))) - set(idx_))
        val_targets = targets[val_idx].tolist()

        target_idx_dict = {}
        for i in range(self.num_classes):
            target_idx_dict[i] = []

        for idx, target in zip(val_idx, val_targets):
            target_idx_dict[target].append(idx)

        val_idx_ = []
        for i in range(self.num_classes):
            val_idx_ = val_idx_ + target_idx_dict[i]
        val_idx = val_idx_

        val_data, val_targets = data[val_idx], targets[val_idx]
        val_data = np.array([x for x in val_data] + [np.fliplr(x) for x in val_data])
        val_targets = np.array(val_targets.tolist() + val_targets.tolist())

        return np.array(data_), targets_, val_data, val_targets

    def longtail(self):
        """
        Apply long-tail sampling to labeled and unlabeled data.
        """
        dataset_class_wise = {}

        for i in range(self.num_classes):
            dataset_class_wise[i] = []

        for i in range(len(self.lb_targets)):
            img, label = self.lb_data[i], self.lb_targets[i]
            dataset_class_wise[label].append(i)

        lamda = math.exp(-1 * math.log(self.imbalance_l) / (self.num_classes - 1))
        for i in range(self.num_classes):
            num_samples = max(int(lamda ** i * self.N1), int(self.N1 / self.imbalance_l))
            dataset_class_wise[i] = dataset_class_wise[i][:num_samples]

        select_list = []
        for i in range(self.num_classes):
            select_list = select_list + dataset_class_wise[i]

        self.lb_data = self.lb_data[np.array(select_list)]
        self.lb_targets = list(self.lb_targets[x] for x in select_list)

    def get_dataset(self):
        """
        Create labeled, unlabeled, and validation datasets.

        Returns:
            Labeled dataset, unlabeled dataset, and validation dataset.
        """
        lb_dset = BasicDataset(self.lb_data, self.lb_targets, self.classes, self.num_classes,
                               self.train_transform, False, None, False)

        val_dset = BasicDataset(self.val_data, self.val_targets, self.classes, self.num_classes,
                                self.test_transform, False, None, False)
        return lb_dset, val_dset