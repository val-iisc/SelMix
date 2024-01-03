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
            preds = self.model.infer_feats(x)
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
   

class MeanRecall(MetricOptimizer):
    def __init__(self, CM, prototypes, model, DistTemp=1, lambda_min=0.6):
        super().__init__(CM, prototypes, model, DistTemp=DistTemp, lambda_min=lambda_min)
        # the rate of gain in the metric of interest
        # if the logit A_{i,j} increases
        self.MetricGainRate = self.metric_gain_rate()
        
        # the sampling distribution and the gain rate
        # in the metric for a given i-j mixup
        self.P, self.G = self.SamplingDistribution()

    def metric_gain_rate(self):
        '''
        returns dM/dA_{i,j}
        '''
        dM_dA = np.zeros((self.num_classes, self.num_classes))
        for i, j in itertools.product(list(range(self.num_classes)),  repeat=2):
            if i==j:
                dM_dA[i, i] = self.CM[i, i] - self.num_classes * (self.CM[i, i] ** 2)
            else:
                dM_dA[i, j] = -1 * self.num_classes * self.CM[i, j] * self.CM[i, i]
        return dM_dA


    def SamplingDistribution(self):
        logits = np.zeros((self.num_classes, self.num_classes))
        MetricGainRate = self.MetricGainRate

        for i, j in itertools.product(list(range(self.num_classes)),  repeat=2):
            logits[i, j] = np.sum(MetricGainRate * self.LogitChangeRate[i, j, :, :])
        
        mask = (logits > 0).astype(float)
        
        P = softmax(logits/self.DistTemp) * mask
        P = (P + 1e-8 )/(np.sum(P + 1e-8 ))
        return P, logits


class MinRecall(MetricOptimizer):
    def __init__(self, CM, prototypes, model, DistTemp=1,\
                 lambdas=None, beta=1.0, val_lr=1.0, lambda_min=0.8):
        super().__init__(CM, prototypes, model, DistTemp=DistTemp, lambda_min=lambda_min)
        self.lambdas = lambdas
        self.beta = beta
        self.val_lr = val_lr
        
        # the rate of gain in the metric of interest
        # if the logit A_{i,j} increases
        self.lambda_update()
        self.MetricGainRate = self.metric_gain_rate()
        
        # the sampling distribution and the gain rate
        # in the metric for a given i-j mixup
        self.P, self.G = self.SamplingDistribution()

    def lambda_update(self):
        '''
        update the lagrange multiplier for the given method
        '''
        print(self.lambdas)
        recall = (np.diag(self.CM)/np.sum(self.CM, 1)).tolist()
        new_lamdas_ = [(x ** self.beta) * np.exp(-1 * self.val_lr * r)\
                        for x, r in zip(self.lambdas, recall)]
        
        self.lambdas = [x/sum(new_lamdas_) for x in new_lamdas_]
        return

    def metric_gain_rate(self):
        '''
        returns dM/dA_{i,j}
        '''
        dM_dA = np.zeros((self.num_classes, self.num_classes))
        for i, j in itertools.product(list(range(self.num_classes)),  repeat=2):
            if i==j:
                dM_dA[i, i] =  self.lambdas[i] * self.num_classes * (self.CM[i, i] - self.num_classes * (self.CM[i, i] ** 2))
            else:
                dM_dA[i, j] = self.lambdas[i] * self.num_classes * (-1 * self.num_classes * self.CM[i, j] * self.CM[i, i])
        return dM_dA

    def SamplingDistribution(self):
        logits = np.zeros((self.num_classes, self.num_classes))
        MetricGainRate = self.MetricGainRate

        for i, j in itertools.product(list(range(self.num_classes)),  repeat=2):
            logits[i, j] = np.sum(MetricGainRate * self.LogitChangeRate[i, j, :, :])

        mask = (logits > 0).astype(float)

        P = softmax(logits/self.DistTemp) * mask
        P = (P + 1e-8 )/(np.sum(P + 1e-8 ))
        return P, logits


class MeanRecallWithCoverage(MetricOptimizer):
    def __init__(self, CM, prototypes, model, DistTemp=1,\
                 lambdas=None, alpha=0.95, tau=0.1, lambda_max=10, lambda_min=0.8):
        super().__init__(CM, prototypes, model, DistTemp=DistTemp, lambda_min=lambda_min)
        self.alpha = alpha
        self.tau = tau
        self.lambda_max = lambda_max 
        self.lambdas = lambdas
        self.lambdas_ = [0.0] * self.num_classes
        self.coverages = np.sum(self.CM, 0)
        self.target_coverage = self.alpha/self.num_classes
        self.coverage_diff = self.coverages - self.target_coverage
        self.lambda_update()

        self.MetricGainRate = self.metric_gain_rate()
        self.P, self.G = self.SamplingDistribution()


    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def lambda_update(self):
        lambdas = [max(self.lambda_max * (1-math.exp(cd/self.tau)), 0) for cd in self.coverage_diff.tolist()]
        print("coverages", self.coverage_diff)

        self.lambdas = lambdas
        print("lambdas", self.lambdas)
        return


    def metric_gain_rate(self):
        '''
        returns dM/dA_{i,j}
        '''
        print("metirc is fixed now")
        dM1_dA = np.zeros((self.num_classes, self.num_classes))
        for i, j in itertools.product(list(range(self.num_classes)),  repeat=2):
            if i==j:
                dM1_dA[i, i] = self.CM[i, i] - self.num_classes * (self.CM[i, i] ** 2)
            else:
                dM1_dA[i, j] = -1 * self.num_classes * self.CM[i, j] * self.CM[i, i]

        dC_dA = np.zeros((self.num_classes, self.num_classes, self.num_classes, self.num_classes))
        for i, j, k, l in itertools.product(list(range(self.num_classes)),  repeat=4):
            if k!=i:
                dC_dA[i,j,k,l] = 0
            else:
                if l == j:
                    dC_dA[i,j,k,l] =  self.CM[i,j] - self.num_classes * (self.CM[i,j] ** 2)
                else:
                    dC_dA[i,j,k,l] = -1 * self.num_classes * self.CM[i,l] * self.CM[i,j]

        dM2_dA = np.zeros((self.num_classes, self.num_classes))
        for i, j, k, l in itertools.product(list(range(self.num_classes)),  repeat=4):
            dM2_dA[k, l] += self.lambdas[j] * dC_dA[i, j, k,l]

        dM_dA = (dM1_dA + dM2_dA)
        return dM_dA/(max(self.lambdas) + 1)



    def SamplingDistribution(self):
        logits = np.zeros((self.num_classes, self.num_classes))
        MetricGainRate = self.MetricGainRate

        for i, j in itertools.product(list(range(self.num_classes)),  repeat=2):
            logits[i, j] = np.sum(MetricGainRate * self.LogitChangeRate[i, j, :, :])

        mask = (logits > 0).astype(float)

        P = softmax(logits/self.DistTemp) * mask
        P = (P + 1e-8 )/(np.sum(P + 1e-8 ))
        return P, logits


class Gmean(MetricOptimizer):
    def __init__(self, CM, prototypes, model, DistTemp=1,  lambda_min=0.6):
        super().__init__(CM, prototypes, model, DistTemp=DistTemp, lambda_min=lambda_min)
        # this is usefull in the expression
        self.CM_rowsum = np.sum(self.CM, 1)

        self.recall = (np.diag(CM)/np.sum(CM, 1)).tolist()
        self.GM = gmean(self.recall)

        # an instantaneous update of the lagrange multiplier
        self.MetricGainRate = self.metric_gain_rate()
        self.P, self.G = self.SamplingDistribution()

    def metric_gain_rate(self):
        '''
        returns dM/dA_{i,j}
        '''
        dlogM_dC = np.zeros((self.num_classes, self.num_classes))
        for i, j in itertools.product(list(range(self.num_classes)),  repeat=2):
            if i==j:
                dlogM_dC[i, j] = (1/self.num_classes) * ( 1/self.CM[i, i] - 1 /(self.CM_rowsum[i]) )
            else:
                dlogM_dC[i, j] = (1/self.num_classes) * ( - 1 /(self.CM_rowsum[i]) )
        
        dM_dC = self.GM * dlogM_dC

        dC_dA = np.zeros((self.num_classes, self.num_classes, self.num_classes, self.num_classes))
        for i, j, l in itertools.product(list(range(self.num_classes)),  repeat=3):
            if l == j:
                dC_dA[i,j,i,l] =self.CM[i,j] - self.num_classes * (self.CM[i,j] ** 2)
            else:
                dC_dA[i,j,i,l] = -1 * self.num_classes * self.CM[i,l] * self.CM[i,j]

        dM_dA = np.zeros((self.num_classes, self.num_classes))
        for i, j in itertools.product(list(range(self.num_classes)),  repeat=2):
             dM_dA += dM_dC[i, j] * dC_dA[i, j, :, :]
        return dM_dA

    def SamplingDistribution(self):
        logits = np.zeros((self.num_classes, self.num_classes))
        MetricGainRate = self.MetricGainRate

        for i, j in itertools.product(list(range(self.num_classes)),  repeat=2):
            logits[i, j] = np.sum(MetricGainRate * self.LogitChangeRate[i, j, :, :])

        mask = (logits > 0).astype(float)

        P = softmax(logits/self.DistTemp) * mask
        P = (P + 1e-8 )/(np.sum(P + 1e-8 ))
        return P, logits


class Hmean(MetricOptimizer):
    def __init__(self, CM, prototypes, model, DistTemp=1, lambda_min=0.6):
        super().__init__(CM, prototypes, model, DistTemp=DistTemp, lambda_min=lambda_min)
        self.recall = (np.diag(CM)/np.sum(CM, 1)).tolist()
        self.CM_rowsum = np.sum(self.CM, axis=1)
        self.hmean = stats.hmean(self.recall)

        # an instantaneous update of the lagrange multiplier
        self.MetricGainRate = self.metric_gain_rate()
        self.P, self.G = self.SamplingDistribution()

    def metric_gain_rate(self):
        '''
        returns dM/dA_{i,j}
        '''
        dM_dC = np.zeros((self.num_classes, self.num_classes))
        for i, j in itertools.product(list(range(self.num_classes)),  repeat=2):
            if i!=j:
                dM_dC[i, j] = (-1 * (self.hmean**2)/self.num_classes) * (1/self.CM[i,i])
            else:
                dM_dC[i, j] = (-1 * (self.hmean**2)/self.num_classes) * (1/self.CM[i,i] - self.CM_rowsum[i]/(self.CM[i,i]**2))

        dC_dA = np.zeros((self.num_classes, self.num_classes, self.num_classes, self.num_classes))
        for i, j, l in itertools.product(list(range(self.num_classes)),  repeat=3):
            if l == j:
                dC_dA[i,j,i,l] =self.CM[i,j] - self.num_classes * (self.CM[i,j] ** 2)
            else:
                dC_dA[i,j,i,l] = -1 * self.num_classes * self.CM[i,l] * self.CM[i,j]

        dM_dA = np.zeros((self.num_classes, self.num_classes))
        for i, j in itertools.product(list(range(self.num_classes)),  repeat=2):
             dM_dA += dM_dC[i, j] * dC_dA[i, j, :, :]
        return dM_dA

    def SamplingDistribution(self):
        logits = np.zeros((self.num_classes, self.num_classes))
        MetricGainRate = self.MetricGainRate

        for i, j in itertools.product(list(range(self.num_classes)),  repeat=2):
            logits[i, j] = np.sum(MetricGainRate * self.LogitChangeRate[i, j, :, :])

        mask = (logits > 0).astype(float)

        P = softmax(logits/self.DistTemp) * mask
        P = (P + 1e-8 )/(np.sum(P + 1e-8 ))
        return P, logits


class HmeanWithCoverage(MetricOptimizer):
    def __init__(self, CM, prototypes, model, DistTemp=1, lambda_min=0.6,\
                 lambdas=None, alpha=0.95, tau=0.1, lambda_max=10):
        super().__init__(CM, prototypes, model, DistTemp=DistTemp, lambda_min=lambda_min)
        self.alpha = alpha
        self.tau = tau
        self.lambda_max = lambda_max 

        # this is usefull in the expression
        self.CM_rowsum = np.sum(self.CM, 1)

        self.coverages = np.sum(self.CM, 0)
        self.target_coverage = self.alpha/self.num_classes
        self.coverage_diff = self.coverages - self.target_coverage

        self.recall = (np.diag(CM)/np.sum(CM, 1)).tolist()
        self.hmean = stats.hmean(self.recall)
        
        self.lambda_update()
        self.MetricGainRate = self.metric_gain_rate()
        self.P, self.G = self.SamplingDistribution()

    def lambda_update(self):
        self.lambdas = [max(self.lambda_max * (1-math.exp(cd/self.tau)), 0)\
                            for cd in self.coverage_diff.tolist()]
        return

    def metric_gain_rate(self):
        '''
        returns dM/dA_{i,j}
        '''
        dM1_dC = np.zeros((self.num_classes, self.num_classes))
        for i, j in itertools.product(list(range(self.num_classes)),  repeat=2):
            if i!=j:
                dM1_dC[i, j] = (-1 * (self.hmean**2)/self.num_classes) * (1/self.CM[i,i])
            else:
                dM1_dC[i, j] = (-1 * (self.hmean**2)/self.num_classes) * (1/self.CM[i,i] - self.CM_rowsum[i]/(self.CM[i,i]**2))

        dC_dA = np.zeros((self.num_classes, self.num_classes, self.num_classes, self.num_classes))
        for i, j, k, l in itertools.product(list(range(self.num_classes)),  repeat=4):
            if k!=i:
                dC_dA[i,j,k,l] = 0
            else:
                if l == j:
                    dC_dA[i,j,k,l] = self.CM[i,j] - self.num_classes * (self.CM[i,j] ** 2)
                else:
                    dC_dA[i,j,k,l] = -1 * self.num_classes * self.CM[i,l] * self.CM[i,j]

        dM1_dA = np.zeros((self.num_classes, self.num_classes))
        for i, j, k, l in itertools.product(list(range(self.num_classes)),  repeat=4):
             dM1_dA[k, l] += dM1_dC[i,j] * dC_dA[i, j, k, l]

        
        
        dM2_dA = np.zeros((self.num_classes, self.num_classes))
        for i, j, k, l in itertools.product(list(range(self.num_classes)),  repeat=4):
            dM2_dA[k, l] += self.lambdas[j] * dC_dA[i, j, k,l]

        dM_dA = (dM1_dA + dM2_dA)
        return dM_dA/(max(self.lambdas) + 1)

    

    def SamplingDistribution(self):
        logits = np.zeros((self.num_classes, self.num_classes))
        MetricGainRate = self.MetricGainRate

        for i, j in itertools.product(list(range(self.num_classes)),  repeat=2):
            logits[i, j] = np.sum(MetricGainRate * self.LogitChangeRate[i, j, :, :])

        mask = (logits > 0).astype(float)

        P = softmax(logits/self.DistTemp) * mask
        P = (P + 1e-8 )/(np.sum(P + 1e-8 ))
        return P, logits


class MinHTRecall(MetricOptimizer):
    def __init__(self, CM, prototypes, model, DistTemp=1,\
                 lambdas=None, beta=1.0, val_lr=1.0, lambda_min=0.8):
        super().__init__(CM, prototypes, model, DistTemp=DistTemp, lambda_min=lambda_min)
        self.lambdas = lambdas
        self.beta = beta
        self.val_lr = val_lr
        
        # the rate of gain in the metric of interest
        # if the logit A_{i,j} increases
        self.lambda_update()
        self.MetricGainRate = self.metric_gain_rate()
        
        # the sampling distribution and the gain rate
        # in the metric for a given i-j mixup
        self.P, self.G = self.SamplingDistribution()

    def lambda_update(self):
        '''
        update the lagrange multiplier for the given method
        '''
        recall = (np.diag(self.CM)/np.sum(self.CM, 1)).tolist()

        head_recall = np.mean(recall[:int(0.9 * self.num_classes)])
        tail_recall = np.mean(recall[int(0.9 * self.num_classes):])

        lambda_h, lambda_t = np.exp(-1 * self.val_lr * head_recall),\
                                np.exp(-1 * self.val_lr * tail_recall)
        lambda_h, lambda_t = lambda_h/(lambda_h + lambda_t), lambda_t/(lambda_h + lambda_t)
        
        self.lambdas = [lambda_h/(0.9 * self.num_classes) if i < 0.9 * self.num_classes\
                            else lambda_t/(0.1 * self.num_classes) for i in range(self.num_classes)]
        
        return

    def metric_gain_rate(self):
        '''
        returns dM/dA_{i,j}
        '''
        dM_dA = np.zeros((self.num_classes, self.num_classes))
        for i, j in itertools.product(list(range(self.num_classes)),  repeat=2):
            if i==j:
                dM_dA[i, i] =  self.lambdas[i] * self.num_classes * (self.CM[i, i] - self.num_classes * (self.CM[i, i] ** 2))
            else:
                dM_dA[i, j] = self.lambdas[i] * self.num_classes * (-1 * self.num_classes * self.CM[i, j] * self.CM[i, i])
        return dM_dA

    def SamplingDistribution(self):
        logits = np.zeros((self.num_classes, self.num_classes))
        MetricGainRate = self.MetricGainRate

        for i, j in itertools.product(list(range(self.num_classes)),  repeat=2):
            logits[i, j] = np.sum(MetricGainRate * self.LogitChangeRate[i, j, :, :])

        mask = (logits > 0).astype(float)

        P = softmax(logits/self.DistTemp) * mask
        P = (P + 1e-8 )/(np.sum(P + 1e-8 ))
        return P, logits


class MeanRecallWithHTCoverage(MetricOptimizer):
    def __init__(self, CM, prototypes, model, DistTemp=1,\
                 lambdas=None, alpha=0.95, tau=0.1, lambda_max=10, lambda_min=0.8):
        super().__init__(CM, prototypes, model, DistTemp=DistTemp, lambda_min=lambda_min)
        self.alpha = alpha
        self.tau = tau
        self.lambda_max = lambda_max 

        self.coverages = np.sum(self.CM, 0)
        self.head_coverage = np.mean(self.coverages[:int(0.9 * self.num_classes)])
        self.tail_coverage = np.mean(self.coverages[int(0.9 * self.num_classes):])

        self.target_coverage = self.alpha/self.num_classes
        self.coverage_diff = self.coverages - self.target_coverage
        self.lambda_update()

        self.MetricGainRate = self.metric_gain_rate()
        self.P, self.G = self.SamplingDistribution()

    def lambda_update(self):
        head_coverage_diff = self.head_coverage - self.target_coverage
        tail_coverage_diff = self.tail_coverage - self.target_coverage
        lambda_h = max(self.lambda_max * (1-math.exp(head_coverage_diff/self.tau)), 0)
        lambda_t = max(self.lambda_max * (1-math.exp(tail_coverage_diff/self.tau)), 0)
        
        self.lambdas = [lambda_h/(0.9 * self.num_classes) if i < 0.9 * self.num_classes\
                            else lambda_t/(0.1 * self.num_classes) for i in range(self.num_classes)]
        self.lambda_h = lambda_h
        self.lambda_t = lambda_t
        print("lambdas", self.lambdas)
        return


    def metric_gain_rate(self):
        '''
        returns dM/dA_{i,j}
        '''
        print("metirc is fixed now")
        dM1_dA = np.zeros((self.num_classes, self.num_classes))
        for i, j in itertools.product(list(range(self.num_classes)),  repeat=2):
            if i==j:
                dM1_dA[i, i] = self.CM[i, i] - self.num_classes * (self.CM[i, i] ** 2)
            else:
                dM1_dA[i, j] = -1 * self.num_classes * self.CM[i, j] * self.CM[i, i]

        dC_dA = np.zeros((self.num_classes, self.num_classes, self.num_classes, self.num_classes))
        for i, j, l in itertools.product(list(range(self.num_classes)),  repeat=3):
            if l == j:
                dC_dA[i,j,i,l] =self.CM[i,j] - self.num_classes * (self.CM[i,j] ** 2)
            else:
                dC_dA[i,j,i,l] = -1 * self.num_classes * self.CM[i,l] * self.CM[i,j]

        dM2_dA = np.zeros((self.num_classes, self.num_classes))

        for i, j in itertools.product(list(range(self.num_classes)),  repeat=2):
            if j < 0.9 * self.num_classes:
                dM2_dA += (self.lambdas[j] * dC_dA[i, j])
            else:
                dM2_dA += (self.lambdas[j] * dC_dA[i, j])

        dM_dA = (dM1_dA + dM2_dA)
        return dM_dA/(max(self.lambda_h, self.lambda_t) + 1)



    def SamplingDistribution(self):
        logits = np.zeros((self.num_classes, self.num_classes))
        MetricGainRate = self.MetricGainRate

        for i, j in itertools.product(list(range(self.num_classes)),  repeat=2):
            logits[i, j] = np.sum(MetricGainRate * self.LogitChangeRate[i, j, :, :])

        mask = (logits > 0).astype(float)

        P = softmax(logits/self.DistTemp) * mask
        P = (P + 1e-8 )/(np.sum(P + 1e-8 ))
        return P, logits


class HmeanWithHTCoverage(MetricOptimizer):
    def __init__(self, CM, prototypes, model, DistTemp=1,\
                 lambdas=None, alpha=0.95, tau=0.1, lambda_max=10, lambda_min=0.8):
        super().__init__(CM, prototypes, model, DistTemp=DistTemp, lambda_min=lambda_min)
        self.alpha = alpha
        self.tau = tau
        self.lambda_max = lambda_max 

        self.recall = (np.diag(self.CM)/np.sum(self.CM, 1)).tolist()
        self.CM_rowsum = np.sum(self.CM, axis=1)
        self.hmean = stats.hmean(self.recall)

        self.coverages = np.sum(self.CM, 0)
        self.head_coverage = np.mean(self.coverages[:int(0.9 * self.num_classes)])
        self.tail_coverage = np.mean(self.coverages[int(0.9 * self.num_classes):])

        self.target_coverage = self.alpha/self.num_classes
        self.coverage_diff = self.coverages - self.target_coverage
        self.lambda_update()

        self.MetricGainRate = self.metric_gain_rate()
        self.P, self.G = self.SamplingDistribution()

    def lambda_update(self):
        head_coverage_diff = self.head_coverage - self.target_coverage
        tail_coverage_diff = self.tail_coverage - self.target_coverage
        lambda_h = max(self.lambda_max * (1-math.exp(head_coverage_diff/self.tau)), 0)
        lambda_t = max(self.lambda_max * (1-math.exp(tail_coverage_diff/self.tau)), 0)
        
        self.lambdas = [lambda_h/(0.9 * self.num_classes) if i < 0.9 * self.num_classes\
                            else lambda_t/(0.1 * self.num_classes) for i in range(self.num_classes)]
        self.lambda_h = lambda_h
        self.lambda_t = lambda_t
        print("lambdas", self.lambdas)
        return


    def metric_gain_rate(self):
        '''
        returns dM/dA_{i,j}
        '''
        dC_dA = np.zeros((self.num_classes, self.num_classes, self.num_classes, self.num_classes))
        for i, j, l in itertools.product(list(range(self.num_classes)),  repeat=3):
            if l == j:
                dC_dA[i,j,i,l] =self.CM[i,j] - self.num_classes * (self.CM[i,j] ** 2)
            else:
                dC_dA[i,j,i,l] = -1 * self.num_classes * self.CM[i,l] * self.CM[i,j]

        dM1_dC = np.zeros((self.num_classes, self.num_classes))
        for i, j in itertools.product(list(range(self.num_classes)),  repeat=2):
            if i!=j:
                dM1_dC[i, j] = (-1 * (self.hmean**2)/self.num_classes) * (1/self.CM[i,i])
            else:
                dM1_dC[i, j] = (-1 * (self.hmean**2)/self.num_classes) * (1/self.CM[i,i] - self.CM_rowsum[i]/(self.CM[i,i]**2))

        dM2_dA = np.zeros((self.num_classes, self.num_classes))

        for i, j in itertools.product(list(range(self.num_classes)),  repeat=2):
            if j < 0.9 * self.num_classes:
                dM2_dA += (self.lambdas[j] * dC_dA[i, j])
            else:
                dM2_dA += (self.lambdas[j] * dC_dA[i, j])


        dM1_dA = np.zeros((self.num_classes, self.num_classes))
        for i, j in itertools.product(list(range(self.num_classes)),  repeat=2):
             dM1_dA += dM1_dC[i, j] * dC_dA[i, j, :, :]
        
        dM_dA = (dM1_dA + dM2_dA)
        return dM_dA/(max(self.lambdas) + 1)


    def SamplingDistribution(self):
        logits = np.zeros((self.num_classes, self.num_classes))
        MetricGainRate = self.MetricGainRate

        for i, j in itertools.product(list(range(self.num_classes)),  repeat=2):
            logits[i, j] = np.sum(MetricGainRate * self.LogitChangeRate[i, j, :, :])

        mask = (logits > 0).astype(float)

        P = softmax(logits/self.DistTemp) * mask
        P = (P + 1e-8 )/(np.sum(P + 1e-8 ))
        return P, logits