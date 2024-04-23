from collections import Counter
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from datasets.index_dataset import IndexDataset


class FastJointSampler:
    """
    A class for joint sampling from two datasets based on a provided sampling distribution.

    Args:
        dataset1 (Dataset): Labeled torch dataset used for MU.
        dataset2 (Dataset): Unlabeled torch dataset.
        model: The model used for generating pseudo-labels.
        samp_dist (numpy.ndarray): A KXK numpy matrix representing P(y1, y2).
        batch_size (int): Size of the batch to be returned.
        semi_supervised (bool): If True, pseudo-labels are generated for dataset2.

    Attributes:
        dataset1 (Dataset): Labeled dataset.
        dataset2 (Dataset): Unlabeled dataset.
        batch_size (int): Batch size.
        model: The model for pseudo-label generation.
        sampling_distribution (numpy.ndarray): Sampling distribution matrix.
        num_classes (int): Number of classes.
    """

    def __init__(self, dataset1, dataset2, model, samp_dist, batch_size=256, semi_supervised=False):
        self.semi_supervised = semi_supervised
        self.dataset1 = deepcopy(dataset1)
        self.dataset2 = deepcopy(dataset2)
        self.batch_size = batch_size
        self.model = model

        self.sampling_distribution = samp_dist
        self.num_classes = samp_dist.shape[0]

        epsilon = 1.0 / (self.num_classes ** 3)
        self.p_y1 = np.sum(self.sampling_distribution + epsilon, axis=1)
        self.p_y2_given_y1 = ((self.sampling_distribution.T + epsilon) / (self.p_y1 + epsilon)).T

        if self.semi_supervised:
            self.update_pseudo_label()
        self.prior_update()

        self.dataset2_idx_dataset = IndexDataset(self.dataset2.targets)

        (
            self.y1_loader,
            self.y1_iter,
            self.y2_given_y1_loader_dict,
            self.y2_given_y1_iter_dict,
        ) = self.get_loaders()

    def prior_update(self):
        """
        Updates the prior probabilities for both datasets based on their targets.
        """
        num_y1 = len(self.dataset1.targets)
        num_y2 = len(self.dataset2.targets)

        prior1 = [Counter(self.dataset1.targets)[i] / num_y1 for i in range(self.num_classes)]
        prior2 = [Counter(self.dataset2.targets)[i] / num_y2 for i in range(self.num_classes)]

        self.dataset1.prior, self.dataset2.prior = prior1, prior2

    def update_pseudo_label(self):
        """
        Updates the pseudo-labels for the unlabeled dataset using the provided model.
        """
        predictions = []
        dataloader2 = DataLoader(self.dataset2, batch_size=self.batch_size, num_workers=8, shuffle=False)
        self.model.eval()

        with torch.no_grad():
            for u_w, _ in dataloader2:
                _, preds = torch.max(self.model(u_w.cuda()), dim=1)
                preds = preds.cpu().tolist()
                predictions += preds

        self.dataset2.targets = predictions

    def get_lb_batch(self):
        """
        Gets a batch from the labeled dataset (dataset1).
        """
        try:
            return next(self.y1_iter)
        except StopIteration:
            self.y1_iter = iter(self.y1_loader)
            return next(self.y1_iter)

    def get_y2_given_y1_sample(self, y1):
        """
        Gets a sample from the unlabeled dataset (dataset2) given a y1 value.
        """
        try:
            return next(self.y2_given_y1_iter_dict[y1])
        except StopIteration:
            self.y2_given_y1_iter_dict[y1] = iter(self.y2_given_y1_loader_dict[y1])
            return next(self.y2_given_y1_iter_dict[y1])

    def get_batch(self):
        """
        Gets a batch containing (x1, y1, x2, y2).
        """
        X2, Y2 = [], []
        X1, Y1 = self.get_lb_batch()

        for i in Y1.numpy().tolist():
            x2_idx, y2 = self.get_y2_given_y1_sample(i)
            x2, y2_ = self.dataset2[x2_idx]
            assert y2 == y2_
            X2.append(x2)
            Y2.append(torch.tensor(y2))

        X2 = torch.stack(X2, dim=0)
        Y2 = torch.stack(Y2, dim=0)
        return X1, Y1, X2, Y2

    def get_loaders(self):
        """
        Generates and returns data loaders for both labeled and unlabeled datasets.
        """
        y1_wts = [float(self.p_y1[y1] / self.dataset1.prior[y1]) for y1 in self.dataset1.targets]

        y2_given_y1_wts = {
            i: [float(self.p_y2_given_y1[y1, y2] / self.dataset2.prior[y2]) for y2 in self.dataset2.targets]
            for i, y1 in enumerate(range(self.num_classes))
        }

        sampler1 = WeightedRandomSampler(weights=y1_wts, num_samples=len(self.dataset1.targets), replacement=True)

        y1_loader = DataLoader(self.dataset1, batch_size=self.batch_size, num_workers=8, sampler=sampler1)
        y1_iter = iter(y1_loader)

        y2_given_y1_loader_dict = {}
        y2_given_y1_iter_dict = {}

        for i in range(self.num_classes):
            sampler = WeightedRandomSampler(weights=y2_given_y1_wts[i], num_samples=len(self.dataset2.targets),
                                            replacement=True)

            loader = DataLoader(self.dataset2_idx_dataset, batch_size=None, num_workers=0, sampler=sampler)
            y2_given_y1_loader_dict[i] = loader
            y2_given_y1_iter_dict[i] = iter(loader)

        return y1_loader, y1_iter, y2_given_y1_loader_dict, y2_given_y1_iter_dict


if __name__ == "__main__":
    # Example usage of FastJointSampler
    print("Add some tests for FastJointSampler")
