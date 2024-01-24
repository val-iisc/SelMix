import itertools
import math
import numpy as np
from scipy.special import softmax
from scipy import stats
from scipy.stats.mstats import gmean

import torch
import torch.nn.functional as F
import multiprocessing


class MetricOptimizer:
    def __init__(self, CM, prototypes, model, DistTemp=1, lambda_min=0.6):
        self.CM = np.clip(CM, a_min=1e-7, a_max=10000)
        self.num_classes = CM.shape[0]
        
        self.prototypes = prototypes
        self.dims = prototypes.shape[1]
        self.DistTemp = DistTemp
        self.lambda_min = lambda_min    
    
        self.model = model
        self.model.eval()

        # apply mixups on the class prototypes 
        # to get the mixedup features
        self.mixup_protypes = self.mixups()
        
        # distribution of confidences on the mixed up
        # features over the label set
        self.mixup_prediction = self.inference()
        
        # change in logit l, for sample of class k
        # when i-j mixup occurs and i it the label
        self.LogitChangeRate = self.logit_change_rate()

    def mixups(self):
        '''
        returns a K X K X dims dimensional array mixup_prototypes
        containing the mixed up features prototype where
        MP[i,j] is the mixed up feature vector of i-th and j-th class
        '''
        mixup_prototypes = np.zeros((self.num_classes,\
                                     self.num_classes,
                                     self.dims))
        mean_lambda = (self.lambda_min + 1.0)/2.0

        for i in list(range(self.num_classes)):
            for j in list(range(self.num_classes)):
                mixup_prototypes[i, j, :] = mean_lambda * self.prototypes[i] +\
                                           (1 - mean_lambda) * self.prototypes[j]
        return mixup_prototypes

    def inference(self):
        '''
        provides the classwise distribution of confidences for each 
        of the mixed up prototype, is a K X K X K dimensional array
        '''
        with torch.no_grad():
            x = torch.tensor(self.mixup_protypes).float().cuda()
            preds = self.model.forward_head(x)
            preds_distribution = F.softmax(preds, dim=-1)
            preds_distribution = preds_distribution.detach().cpu().numpy()
        return preds_distribution

    def logit_change_rate(self):
        '''
        returns the (\DeltaW)_{i-j mixup}^{T} Z_q}_l
        a [i,j,q,l] tensor
        we will call it the logit change rate
        '''
        LCR = torch.zeros((self.num_classes, self.num_classes,\
                                      self.num_classes, self.num_classes)).cuda()
        mixup_protypes = torch.tensor(self.mixup_protypes).cuda()
        prototypes = torch.tensor(self.prototypes).cuda()
        mixup_prediction = torch.tensor(self.mixup_prediction).cuda()
        prototypes_dotpdt = torch.zeros((self.num_classes, self.num_classes, self.num_classes)).cuda()
        
        for q in range(self.num_classes):
            prototypes_dotpdt[:,:,q] =  torch.sum(mixup_protypes * prototypes[q], dim=-1)
        
        for q, l in itertools.product(list(range(self.num_classes)),  repeat=2):
            LCR[:, :, q, l] =  -1 * prototypes_dotpdt[:,:,q] * (mixup_prediction[:, :, l])
        for i, q in itertools.product(list(range(self.num_classes)),  repeat=2):
            LCR[i, :, q, i] += prototypes_dotpdt[i,i,q]  
        return LCR.cpu().numpy()