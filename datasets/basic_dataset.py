from torchvision import  transforms
from torch.utils.data import Dataset
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
                 targets=[0,1,2],
                 classes=["cat", "dog", "ship"],
                 num_classes=3,
                 transform=None,
                 use_strong_transform=False,
                 strong_transform=None):
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
                return img_w, self.strong_transform(img), target # type: ignore


    def __len__(self):
        return len(self.data)