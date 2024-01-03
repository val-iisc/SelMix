from torchvision import datasets, transforms
from torch.utils.data import Dataset
from .data_utils import get_onehot
from .augmentation.randaugment import RandAugment

from PIL import Image
from collections import Counter
import numpy as np
import math
import copy


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
        if use_strong_transform:
            if strong_transform is None:
                self.strong_transform = copy.deepcopy(transform)
                self.strong_transform.transforms.insert(0, RandAugment(3,5))
        else:
            self.strong_transform = strong_transform
        self.is_longtail = False
        self.total_samples = sum(Counter(self.targets).values())
        self.prior = [n/self.total_samples for n in Counter(self.targets).values()]

    def longtail(self, imbalance, N1):
        if self.is_longtail:
            print(" [ WARNING ] This dataset has already been made long tail")
            print(" [ WARNING ] proceeding further anyways")

        dataset_class_wise = {}
        dataset = zip(self.data, self.targets)

        for i in range(self.num_classes):
            dataset_class_wise[i] = []

        for i, (img, label) in enumerate(dataset):
            dataset_class_wise[label].append(i)

        lamda = math.exp(-1 * math.log(imbalance)/(self.num_classes - 1))
        for i in range(self.num_classes):
            num_samples = max(int(lamda**i * N1), 1)
            dataset_class_wise[i] = dataset_class_wise[i][:num_samples]
            print(len(dataset_class_wise[i]))

        if num_samples == 1:
            print(" [ WARNING ]  There were far too few samples in this dataset")
            print(" [ WARNING ]  Please either increase N1 or decrease imbalance \n" )

        select_list = []
        for i in range(self.num_classes):
            select_list = select_list + dataset_class_wise[i]
        
        self.data = self.data[np.array(select_list)]
        self.targets = list(self.targets[x] for x in select_list)
        self.total_samples = sum(Counter(self.targets).values())
        self.prior = [n/self.total_samples for n in Counter(self.targets).values()]
        self.is_longtail = True
        return 


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
            
        img = self.data[idx]
        if self.transform is None:
            return transforms.ToTensor()(img), target
        else:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            img_w = self.transform(img)
            if not self.use_strong_transform:
                return img_w, target
            else:
                return img_w, self.strong_transform(img), target

    
    def __len__(self):
        return len(self.data)