import math
import os
from pickletools import optimize
import time

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard.writer import SummaryWriter

import wandb


class TBLog:
    """
    Construc tensorboard writer (self.writer).
    The tensorboard is saved at os.path.join(tb_dir, file_name).
    """
    def __init__(self, tb_dir, file_name):
        self.tb_dir = tb_dir
        self.writer = SummaryWriter(os.path.join(self.tb_dir, file_name))
    
    def update(self, tb_dict, it, suffix=None):
        """
        Args
            tb_dict: contains scalar values for updating tensorboard
            it: contains information of iteration (int).
            suffix: If not None, the update key has the suffix.
        """
        if suffix is None:
            suffix = ''
        
        for key, value in tb_dict.items():
            self.writer.add_scalar(suffix+key, value, it)         

            
class AverageMeter(object):
    """
    refer: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

        
def get_SGD(net, name='SGD', lr=0.1, momentum=0.9, \
                  weight_decay=5e-4, nesterov=True, bn_wd_skip=True):
    '''
    return optimizer (name) in torch.optim.
    If bn_wd_skip, the optimizer does not apply
    weight decay regularization on parameters in batch normalization.
    '''
    optim = getattr(torch.optim, name)
    
    decay = []
    no_decay = []
    for name, param in net.named_parameters():
        if ('bn' in name) and bn_wd_skip:
            no_decay.append(param)
        else:
            decay.append(param)
    
    per_param_args = [{'params': decay},
                      {'params': no_decay, 'weight_decay': 0.0}]
    
    optimizer = optim(per_param_args, lr=lr,
                    momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
    return optimizer
        

def get_finetune_SGD(net, opt='SGD', lr=0.001, weight_decay=5e-4, freeze_backbone=False):
    '''
    return optimizer (name) in torch.optim.
    If bn_wd_skip, the optimizer does not apply
    weight decay regularization on parameters in batch normalization.
    '''
    decay = []
    no_decay = []
    fc = []
    fc_no_decay = []
    for name, param in net.named_parameters():
        if 'fc' in name:
            if 'bn' in name:
                param.requires_grad=False
            if 'bias' in name:
                fc_no_decay.append(param)
            else:
                fc.append(param)
        else:
            if freeze_backbone:
                param.requires_grad=False
            if 'bn' in name:
                param.requires_grad=False
                no_decay.append(param)
                pass
            else:
                decay.append(param)
    if freeze_backbone:
        per_param_args = [{'params': fc, "lr": lr},
                          {'params': fc_no_decay, 'weight_decay': 0.0, "lr": lr}]
    else:
        per_param_args = [{'params': decay, "lr":  lr/10.0},
                          {'params': no_decay, 'weight_decay': 0.0, "lr": 0.0},
                          {'params': fc, "lr": lr},
                          {'params': fc_no_decay, 'weight_decay': 0.0, "lr": lr}]
    
    if opt == "SGD":
        optimizer = torch.optim.SGD(per_param_args, lr=lr, weight_decay=weight_decay)
    elif opt == "Adam":
        print("USing ADAM optimizer")
        optimizer = torch.optim.Adam(per_param_args, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay)
    return optimizer


def get_CRT_SGD(net, name='SGD', lr=0.1, momentum=0.9, \
                  weight_decay=5e-4, nesterov=True, bn_wd_skip=True):
    '''
    return optimizer (name) in torch.optim.
    If bn_wd_skip, the optimizer does not apply
    weight decay regularization on parameters in batch normalization.
    '''
    optim = getattr(torch.optim, name)

    decay = []
    no_decay = []
    for name, param in net.named_parameters():
        if 'fc' in name:
            if ('bn' in name) and bn_wd_skip:
                no_decay.append(param)
            else :
                decay.append(param)
        else:
            param.requires_grad = False

    per_param_args = [{'params': decay},
                      {'params': no_decay, 'weight_decay': 0.0}]

    optimizer = optim(per_param_args, lr=lr,
                    momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
    return optimizer

def get_cosine_schedule_with_warmup(optimizer,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    num_warmup_steps=0,
                                    last_epoch=-1):
    '''
    Get cosine scheduler (LambdaLR).
    if warmup is needed, set num_warmup_steps (int) > 0.
    '''
    
    def _lr_lambda(current_step):
        '''
        _lr_lambda returns a multiplicative factor given an interger parameter epochs.
        Decaying criteria: last_epoch
        '''
        
        if current_step < num_warmup_steps:
            _lr = float(current_step) / float(max(1, num_warmup_steps))
        else:
            num_cos_steps = float(current_step - num_warmup_steps)
            num_cos_steps = num_cos_steps / float(max(1, num_training_steps - num_warmup_steps))
            _lr = max(0.0, math.cos(math.pi * num_cycles * num_cos_steps))
        return _lr
    
    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    
    Args
        output: logits or probs (num of batch, num of classes)
        target: (num of batch, 1) or (num of batch, )
        topk: list of returned k
    
    refer: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    
    with torch.no_grad():
        maxk = max(topk) #get k in top-k
        batch_size = target.size(0) #get batch size of target

        # torch.topk(input, k, dim=None, largest=True, sorted=True, out=None)
        # return: value, index
        _, pred = output.topk(k=maxk, dim=1, largest=True, sorted=True) # pred: [num of batch, k]
        pred = pred.t() # pred: [k, num of batch]
        
        #[1, num of batch] -> [k, num_of_batch] : bool
        correct = pred.eq(target.view(1, -1).expand_as(pred)) 

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        #np.shape(res): [k, 1]
        return res 

    
def ce_loss(logits, targets, use_hard_labels=True, reduction='none'):
    """
    wrapper for cross entropy loss in pytorch.
    
    Args
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
    """
    if use_hard_labels:
        return F.cross_entropy(logits, targets, reduction=reduction)
    else:
        assert logits.shape == targets.shape
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets*log_pred, dim=1)
        return nll_loss

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_samp_dist(CM, classes, title="Chart"):
    fig, ax = plt.subplots()
    im = ax.imshow(CM)
    cbar = ax.figure.colorbar(im, ax = ax)
    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(classes)), labels=classes)
    ax.set_yticks(np.arange(len(classes)), labels=classes)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(classes)):
        for j in range(len(classes)):
            text = ax.text(j, i, round(CM[i, j],2),
                        ha="center", va="center", color="w")

    ax.set_title(title)
    fig.tight_layout()
    wandb.log({title: plt})
    plt.close()


import torch
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.optimizer import Optimizer

class WarmupMultiStepLR(MultiStepLR):
    def __init__(self, optimizer: Optimizer, milestones: list, gamma=0.1, warmup_iters=0, warmup_factor=1e-4):
        self.warmup_iters = warmup_iters
        self.warmup_factor = warmup_factor
        super(WarmupMultiStepLR, self).__init__(optimizer, milestones, gamma)

    def get_lr(self):
        if self.last_epoch < self.warmup_iters:
            # Warmup phase: gradually increase learning rate
            alpha = float(self.last_epoch) / float(max(1, self.warmup_iters))
            factor = self.warmup_factor * (1 - alpha) + alpha
            return [base_lr * factor for base_lr in self.base_lrs]
        else:
            return super(WarmupMultiStepLR, self).get_lr()