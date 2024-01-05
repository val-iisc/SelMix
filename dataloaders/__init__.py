from collections import Counter
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from datasets.index_dataset import IndexDataset

class FastJointSampler:
    def __init__(self, dataset1, dataset2, model, samp_dist, batch_size=256, semi_supervised=False):
        '''
        Returns an object that returns a batch of
        (x1, y1, x2, y2) that satisfy the sampling distribution
        provided i.e P(y1, y2)
        Args:
            dataset1    : a torch dataset object whose label
                        will be used for MU
            dataset2    : a torch dataset object whose label
                        will not be used for MU
            prototypes  : validation set, a torch dataset object
            samp_dist   : a KXK numpy matrix representing P(y1, y2)
            SSL         : if the second dataset needs the pseudo-labels
                          to be generated else use the targets in the datasets
                          to keep the sample in the dataset
            batch_size  : size of the batch to be returned
        '''
        super(FastJointSampler, self).__init__()

        self.semiSupervised = semi_supervised
        self.dataset1 = deepcopy(dataset1)
        self.dataset2 = deepcopy(dataset2)
        self.batch_size = batch_size

        # model is needed to generate pseudo-labels
        self.model = model

        self.sampling_distribution = samp_dist
        self.num_classes = samp_dist.shape[0]
        
        # have a small epsilon for non zero sampling probability 
        # since torch weighted sampler can't handle zero Weights
        epsilon = 1.0/(self.num_classes**3)
        self.p_y1 = np.sum(self.sampling_distribution + epsilon, axis=1)
        self.p_y2_given_y1 = ((self.sampling_distribution.T + epsilon) / (self.p_y1 + epsilon)).T

        # update the pseudolabel of the unlabeled dataset for SemiSL
        # and get the priors for both the datasets
        if self.semiSupervised:
            self.updatePseudolabel()
        self.priorUpdate()

        # create a simple dataset of indexes and targetsso that
        # we don't have to create multiple copies of the same 
        # dataset for y2|y1 data loaders
        self.dataset2_idx_dataset = IndexDataset(self.dataset2)

        self.y1_loader, self.y1_iter, self.y2_given_y1_loader_dict, self.y2_given_y1_iter_dict = self.get_loaders()
        print("All loaders are loaded...")

    def priorUpdate(self):
        total_y1_samples = len(self.dataset1.targets)
        total_y2_samples = len(self.dataset2.targets)

        prior1 = [Counter(self.dataset1.targets)[i] / total_y1_samples for i in range(self.num_classes)]
        prior2 = [Counter(self.dataset2.targets)[i] / total_y2_samples for i in range(self.num_classes)]

        self.dataset1.prior1, self.dataset2.prior2 = prior1, prior2
        return

    def updatePseudolabel(self):
        predictions = []
        dataloader2 = DataLoader(self.dataset2, batch_size=1024, num_workers=8, shuffle=False)
        self.model.eval()

        with torch.no_grad():
            for u_w, _ in dataloader2:
                _, preds = torch.max(self.model(u_w.cuda()), dim=1)
                preds = preds.cpu().tolist()
                predictions += preds

        # we add an attribute target to the second dataset and 
        # set it to the pseudo-labels
        self.dataset2.targets = predictions
        return

    def get_lb_batch(self):
        try:
            return self.y1_iter.next()
        except:
            self.y1_iter = iter(self.y1_loader)
            return self.y1_iter.next()

    def get_y2_given_y1_sample(self, y1):
        try:
            return self.y2_given_y1_iter_dict[y1].next()
        except:
            self.y2_given_y1_iter_dict[y1] = iter(self.y2_given_y1_loader_dict[y1])
            return self.y2_given_y1_iter_dict[y1].next()

    def get_batch(self):
        X2, Y2 = [], []
        X1, Y1 = self.get_lb_batch()

        for i in Y1.numpy().tolist():
            x2_idx, y2 = next(self.y2_given_y1_iter_dict[f'{i}'])
            x2, y2_ = self.dataset2[x2_idx]
            assert y2 == y2_
            X2.append(x2)
            Y2.append(torch.tensor(y2))

        X2 = torch.stack(X2, dim=0)
        Y2 = torch.stack(Y2, dim=0)
        return X1, Y1, X2, Y2

    def get_loaders(self):
        y1_wts = [float(self.p_y1[y1] / self.prior1[y1]) for y1 in self.dataset1.targets]

        y2_given_y1_wts = {i: [float(self.p_y2_given_y1[y1, y2] / self.prior2[y2])
                               for y2 in self.dataset2.targets] for i, y1 in enumerate(range(self.num_classes))}

        sampler1 = WeightedRandomSampler(weights=y1_wts, num_samples=len(self.dataset1.targets), replacement=True)
        if self.random:
            sampler1 = None
        y1_loader = DataLoader(self.dataset1, batch_size=self.batch_size, num_workers=8, sampler=sampler1)
        y1_iter = iter(y1_loader)

        y2_given_y1_loader_dict = {}
        y2_given_y1_iter_dict = {}

        for i in range(self.num_classes):
            print("generating loader : ", i)
            sampler = WeightedRandomSampler(weights=y2_given_y1_wts[i], num_samples=len(self.dataset2.targets),
                                            replacement=True)
            if self.random:
                sampler = None
            loader = DataLoader(self.dataset2_idx_dataset, batch_size=None, num_workers=0, sampler=sampler)
            y2_given_y1_loader_dict.update({f"{i}": loader})
            y2_given_y1_iter_dict.update({f"{i}": iter(loader)})

        return y1_loader, y1_iter, y2_given_y1_loader_dict, y2_given_y1_iter_dict

if __name__ == "__main__":
    print('work in progress')
