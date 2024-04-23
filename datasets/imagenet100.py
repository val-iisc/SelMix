import numpy as np
import math
from collections import Counter
import copy
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import random
from torchvision import transforms
from .augmentation.randaugment import RandAugment
import os

class DualAugmentationTransform:
    def __init__(self, weak_transform, strong_transform):
        """
        Initialize DualAugmentationTransform.

        Args:
            weak_transform: Transform for weak augmentation.
            strong_transform: Transform for strong augmentation.
        """
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform

    def __call__(self, img):
        """
        Apply dual augmentations on the input image.

        Args:
            img (PIL Image): Input image.

        Returns:
            tuple: A tuple containing two augmented images.
        """
        # Apply weak augmentation
        weak_img = self.weak_transform(img)

        # Apply strong augmentation
        strong_img = self.strong_transform(img)

        # Return tuple of augmented images
        return weak_img, strong_img

def get_weak_transform(mean, std, train=True, size=224):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # Add this line to convert PIL image to tensor
            transforms.Normalize(mean, std)
        ])
    else:
        return transforms.Compose([
            transforms.Resize(size + 32),
            transforms.CenterCrop(size),
            transforms.ToTensor(),  # Add this line to convert PIL image to tensor
            transforms.Normalize(mean, std)
        ])

class ImageNet100_SSL_LT_Dataset:
    def __init__(self, data_dir, num_classes=100, N1=433, M1=866, imbalance_l=10.0, imbalance_u=10.0, use_strong_transform=False):
        """
        Initialize ImageNet SSL (Semi-Supervised Learning) Dataset.

        Args:
            data_dir (str): Directory where ImageNet data is stored.
            num_classes (int): Number of classes in the dataset.
            N1 (int): Number of labeled samples.
            M1 (int): Number of unlabeled samples.
            imbalance_l (float): Imbalance factor for labeled data.
            imbalance_u (float): Imbalance factor for unlabeled data.
        """
        self.data_dir = data_dir
        assert os.path.join(self.data_dir, "train")
        assert os.path.join(self.data_dir, "val")
        self.train_data_dir = os.path.join(self.data_dir, "train")
        self.test_data_dir = os.path.join(self.data_dir, "val")

        self.num_classes = num_classes
        self.N1 = N1
        self.M1 = M1
        self.imbalance_l = imbalance_l
        self.imbalance_u = imbalance_u
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.use_strong_transform = use_strong_transform

        # Define data transformations for training and testing
        self.weak_train_transform = get_weak_transform(self.mean, self.std, train=True, size=224)
        self.strong_train_transform = get_weak_transform(self.mean, self.std, train=True, size=224)
        self.strong_train_transform.transforms.insert(0, RandAugment(3,5))
        
        self.lb_transform = self.weak_train_transform
        self.ulb_transform = DualAugmentationTransform(self.weak_train_transform, self.strong_train_transform)

        self.test_transform = get_weak_transform(self.mean, self.std, train=False, size=224)

        # Load and preprocess the dataset
        self.load_data()
        self.apply_longtail_sampling()
        self.split_test_data()

    def load_data(self):
        """
        Load and preprocess the ImageNet dataset.
        """
        self.train_dataset1 = datasets.ImageFolder(self.train_data_dir, transform=self.lb_transform)
        self.train_dataset2 = datasets.ImageFolder(self.train_data_dir, transform=self.ulb_transform)
        self.test_dataset = datasets.ImageFolder(self.test_data_dir, transform=self.test_transform)

    def apply_longtail_sampling(self):
        """
        Apply long-tail sampling to create labeled and unlabeled datasets.
        """
        targets = np.array(self.train_dataset1.targets)
        lamda_l = math.exp(-1 * math.log(self.imbalance_l) / (self.num_classes - 1))
        lamda_u = math.exp(-1 * math.log(self.imbalance_u) / (self.num_classes - 1))

        lb_samples = []
        ulb_samples = []

        lb_prior = []
        ulb_prior = []

        for i in range(self.num_classes):
            class_indices = np.where(targets == i)[0]
            num_lb_samples = max(int(self.N1 * (lamda_l ** i)), 1)
            num_ulb_samples = max(int(self.M1 * (lamda_u ** i)), 1)
            lb_samples.extend(class_indices[:num_lb_samples])
            ulb_samples.extend(class_indices[:num_ulb_samples])
            lb_prior.append(num_lb_samples)
            ulb_prior.append(num_ulb_samples)

        self.lb_dataset = self.train_dataset1
        self.ulb_dataset = self.train_dataset2
        # Mutate the labeled dataset
        self.lb_dataset.samples = [self.train_dataset1.samples[idx] for idx in lb_samples]
        self.lb_dataset.targets = [self.train_dataset1.targets[idx] for idx in lb_samples]
        self.lb_dataset.imgs = self.lb_dataset.samples
        self.lb_dataset.prior = [x/sum(lb_prior) for x in lb_prior]
        self.lb_dataset.prior = [x/sum(ulb_prior) for x in ulb_prior]

        # Mutate the unlabeled dataset
        self.ulb_dataset.samples = [self.train_dataset2.samples[idx] for idx in ulb_samples]
        self.ulb_dataset.targets = [self.train_dataset2.targets[idx] for idx in ulb_samples]
        self.ulb_dataset.imgs = self.ulb_dataset.samples

    def split_test_data(self, eval_frac=0.5, random_seed=0):
        """
        Split the test dataset into evaluation (eval) and validation (val) datasets.

        Args:
            eval_frac (float): Fraction of the test dataset to use for evaluation.
            random_seed (int): Random seed for reproducibility.
        """
        num_test_samples = len(self.test_dataset)
        eval_size = int(num_test_samples * eval_frac)
        val_size = num_test_samples - eval_size

        indices = list(range(num_test_samples))
        if random_seed is not None:
            random.seed(random_seed)
        random.shuffle(indices)

        eval_indices = indices[:eval_size]
        val_indices = indices[eval_size:]

        self.eval_dataset = copy.deepcopy(self.test_dataset)
        self.eval_dataset.samples = [self.test_dataset.samples[idx] for idx in eval_indices]
        self.eval_dataset.targets = [self.test_dataset.targets[idx] for idx in eval_indices]
        self.eval_dataset.imgs = self.eval_dataset.samples

        self.val_dataset = copy.deepcopy(self.test_dataset)
        self.val_dataset.samples = [self.test_dataset.samples[idx] for idx in val_indices]
        self.val_dataset.targets = [self.test_dataset.targets[idx] for idx in val_indices]
        self.val_dataset.imgs = self.val_dataset.samples

        print(f"Test dataset split into eval: {len(self.eval_dataset)} samples, val: {len(self.val_dataset)} samples")

    def return_splits(self):
        """
        Return the labeled, unlabeled, evaluation, and validation datasets.

        Returns:
            tuple: A tuple containing the labeled, unlabeled, evaluation, and validation datasets.
        """
        return self.lb_dataset, self.ulb_dataset, self.val_dataset, self.eval_dataset
