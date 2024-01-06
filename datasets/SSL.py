import numpy as np
import copy 
import math
from collections import Counter
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler

from .augmentation.randaugment import RandAugment

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
                self.strong_transform.transforms.insert(0, RandAugment(3,5)) # type: ignore
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
                return  img_w, self.strong_transform(img), target # type: ignore

    
    def __len__(self):
        return len(self.data)

class SSL_Dataset:
    """
    SSL_Dataset class gets dataset (cifar10, cifar100) from torchvision.datasets,
    separates labeled and unlabeled data,
    and returns BasicDataset: torch.utils.data.Dataset (see datasets.dataset.py)
    """

    def __init__(self, name='cifar100', train=True, num_classes=100, data_dir='./data',
                 N1=1500, M1=3000, size=45000, include_train=False, uratio=2.0,
                 imbalance_l=100.0, imbalance_u=100.0, use_strong_transform=False,
                 inverted=False, frac=0.0):
        """
        Initializes the SSL_Dataset.

        Args:
            name: Name of the dataset in torchvision.datasets (cifar10, cifar100).
            train: True means the dataset is the training dataset (default=True).
            num_classes: Number of label classes.
            data_dir: Path of the directory where data is downloaded or stored.
            N1: Some parameter (assumed to be 1500 in this example).
            M1: Some parameter (assumed to be 3000 in this example).
            size: Size of the dataset (assumed to be 45000 in this example).
            include_train: Whether to include labeled data in unlabeled data.
            uratio: Some parameter (assumed to be 2.0 in this example).
            imbalance_l: Imbalance parameter for labeled data (assumed to be 100.0 in this example).
            imbalance_u: Imbalance parameter for unlabeled data (assumed to be 100.0 in this example).
            use_strong_transform: Whether to use strong data augmentation.
            inverted: Inversion parameter (assumed to be False in this example).
            frac: Some fraction parameter (assumed to be 0.0 in this example).
        """
        self.mean, self.std = {}, {}
        self.mean['cifar10'] = [x / 255 for x in [125.3, 123.0, 113.9]]
        self.mean['cifar100'] = [x / 255 for x in [129.3, 124.1, 112.4]]
        self.mean['stl10'] = [x / 255 for x in [112.3, 108.9,  98.3]]

        self.std['cifar10'] = [x / 255 for x in [63.0, 62.1, 66.7]]
        self.std['cifar100'] = [x / 255 for x in [68.2,  65.4,  70.4]]
        self.std['stl10'] = [x / 255 for x in [68.5, 66.6, 68.4]]
        
        self.inverted = inverted
        self.name = name
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.N1 = N1
        self.M1 = M1
        self.frac = frac
        self.size = size
        self.include_train = include_train
        self.uratio = uratio
        self.imbalance_l = imbalance_l
        self.imbalance_u = imbalance_u
        self.train_transform, self.val_trainsform = self.get_transform()
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
        print("ULB : ", Counter(self.ulb_targets))
        print("LB : ", Counter(self.lb_targets))
        self.lb_dset, self.ulb_dset, self.val_dset = self.basic_dataset()

    def get_transform(self):
        """
        Gets the transformation for the dataset.

        Args:
            mean: Mean values for normalization.
            std: Standard deviation values for normalization.
            train: Whether the transformation is for training.
            image_size: Size of the image.

        Returns:
            transforms.Compose: Composition of transformations.
        """
        return (transforms.Compose([transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop(self.size, padding=4),
                                    transforms.ToTensor(), 
                                    transforms.Normalize(self.mean, self.std)]),
                transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize(self.mean, self.std)]))

    def get_data(self):
        """
        Gets data from torchvision.datasets.
        """
        if 'cifar' in self.name:
            dset = getattr(torchvision.datasets, self.name.upper())
            dset = dset(self.data_dir, train=True, download=True)
            data, targets, classes = dset.data, dset.targets, dset.classes
            self.data, self.targets, self.val_data, self.val_targets = self.resize(data, targets)
            self.classes = classes
        

    def resize(self, data, targets):
        """
        Resize the dataset based on the specified size and create an augmented validation set.

        Args:
            data (numpy.ndarray): Input data.
            targets (list): List of target labels.

        Returns:
            tuple: Resized data, resized targets, augmented validation data, augmented validation targets.
        """
        
        targets = np.array(targets)
        samples_per_class = self.size // self.num_classes
        data_, targets_, idx_ = [], [], []

        class_indices = {i: [] for i in range(self.num_classes)}

        # Select samples for each class based on the specified size
        for c in range(self.num_classes):
            idx = np.where(targets == c)[0]
            idx = idx[:samples_per_class]
            idx_.extend(idx)
            data_.extend(data[idx])
            targets_.extend(targets[idx])
            class_indices[c].extend(idx)

        val_idx = list(set(range(len(targets))) - set(idx_))
        val_targets = targets[val_idx].tolist()

        val_indices_dict = {i: [] for i in range(self.num_classes)}

        # Store validation set indices for each class
        for idx, target in zip(val_idx, val_targets):
            val_indices_dict[target].append(idx)

        val_idx_ = [val_indices_dict[i][:int(self.frac * len(val_indices_dict[i]))] for i in range(self.num_classes)]
        val_idx = [idx for sublist in val_idx_ for idx in sublist]

        # Create augmented validation set by flipping horizontally
        print("Flipping validation data horizontally")
        val_data, val_targets = data[val_idx], targets[val_idx]
        val_data = np.array([x for x in val_data] + [np.fliplr(x) for x in val_data])
        print("Shape is:", val_data.shape)
        val_targets = np.array(val_targets.tolist() + val_targets.tolist())

        return (np.array(data_), targets_, val_data, val_targets)


    def split(self):
        """
        Split the dataset into labeled and unlabeled parts.

        Returns:
            tuple: Labeled data, labeled targets, unlabeled data, unlabeled targets.
        """
        lb_data = []
        lb_targets = []
        lb_indices = []

        targets = np.array(self.targets)
        samples_per_class = int((len(self.targets) / (self.uratio + 1)) // self.num_classes)

        # Select labeled samples for each class
        for c in range(self.num_classes):
            idx = np.where(targets == c)[0]
            idx = np.random.choice(idx, samples_per_class, replace=False)
            lb_indices.extend(idx)
            lb_data.extend(self.data[idx])
            lb_targets.extend(targets[idx])

        lb_data, lb_targets, lb_indices = np.array(lb_data), np.array(lb_targets), np.array(lb_indices)

        # Unlabeled data is the remaining data after excluding labeled indices
        ulb_indices = np.array(sorted(set(range(len(self.data))) - set(lb_indices)))
        ulb_data = self.data[ulb_indices]
        ulb_targets = targets[ulb_indices].tolist()

        return lb_data, lb_targets, ulb_data, ulb_targets


    def drop_elements_randomly(self, input_list, p):
        """
        Drops elements randomly from the input list.

        Args:
            input_list: List to drop elements from.
            p: Fraction of elements to drop.

        Returns:
            list: Resulting list after dropping elements.
        """
        # Your implementation of drop_elements_randomly
        # ...

    def longtail(self):
        """
        Applies the long tail distribution on the dataset
        """
        def update_dataset(data, targets, num_classes, num_samples, imbalance_factor):
            dataset_class_wise = {i: [] for i in range(num_classes)}

            for i in range(len(targets)):
                img, label = data[i], targets[i]
                dataset_class_wise[label].append(i)

            lamda = math.exp(-1 * math.log(imbalance_factor) / (num_classes - 1))
            for i in range(num_classes):
                num_samples_i = max(int(lamda**i * num_samples), int(num_samples / imbalance_factor))
                dataset_class_wise[i] = dataset_class_wise[i][:num_samples_i]

            select_list = [idx for sublist in dataset_class_wise.values() for idx in sublist]

            return np.array(data[select_list]), [targets[x] for x in select_list]

        # Update labeled dataset
        self.lb_data, self.lb_targets = update_dataset(self.lb_data, self.lb_targets, self.num_classes, self.N1, self.imbalance_l)

        # Update unlabeled dataset
        self.ulb_data, self.ulb_targets = update_dataset(self.ulb_data, self.ulb_targets, self.num_classes, self.M1, self.imbalance_u)
        return 

    def basic_dataset(self):
        """
        Creates BasicDataset instances for labeled, unlabeled, and validation datasets.

        Returns:
            tuple: Labeled dataset, unlabeled dataset, and validation dataset.
        """
        lb_dset = BasicDataset(self.lb_data, self.lb_targets, self.classes, self.num_classes,
                               self.train_transform, False, None, False)

        ulb_dset = BasicDataset(self.ulb_data, self.ulb_targets, self.classes, self.num_classes,
                                self.train_transform, self.use_strong_transform, None, False)

        val_dset = BasicDataset(self.val_data, self.val_targets, self.classes, self.num_classes,
                                self.val_trainsform, False, None, False)
        return lb_dset, ulb_dset, val_dset