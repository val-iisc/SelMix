import os
import time
from torch.utils.tensorboard import SummaryWriter
import logging

def setattr_cls_from_kwargs(cls, kwargs):
    #if default values are in the cls,
    #overlap the value by kwargs
    for key in kwargs.keys():
        if hasattr(cls, key):
            print(f"{key} in {cls} is overlapped by kwargs: {getattr(cls,key)} -> {kwargs[key]}")
        setattr(cls, key, kwargs[key])

        
def test_setattr_cls_from_kwargs():
    class _test_cls:
        def __init__(self):
            self.a = 1
            self.b = 'hello'
    test_cls = _test_cls()
    config = {'a': 3, 'b': 'change_hello', 'c':5}
    setattr_cls_from_kwargs(test_cls, config)
    for key in config.keys():
        print(f"{key}:\t {getattr(test_cls, key)}")
        
        
def net_builder(net_name, from_name: bool, net_conf=None):
    """
    return **class** of backbone network (not instance).
    Args
        net_name: 'WideResNet' or network names in torchvision.models
        from_name: If True, net_buidler takes models in torch.vision models. Then, net_conf is ignored.
        net_conf: When from_name is False, net_conf is the configuration of backbone network (now, only WRN is supported).
    """
    if from_name:
        import torchvision.models as models
        model_name_list = sorted(name for name in models.__dict__
                                if name.islower() and not name.startswith("__")
                                and callable(models.__dict__[name]))

        if net_name not in model_name_list:
            assert Exception(f"[!] Networks\' Name is wrong, check net config, \
                               expected: {model_name_list}  \
                               received: {net_name}")
        else:
            return models.__dict__[net_name]
        
    else:
        if net_name == 'WideResNet':
            import models.nets.wrn as net
            builder = getattr(net, 'build_WideResNet')()
        
        if net_name == 'WideResNet_FM':
            import models.nets.wrn_FM as net
            builder = getattr(net, 'build_WideResNet')()
        if net_name == 'WideResNet_ABC':
            import models.nets.wrn_ABC as net
            builder = getattr(net, 'build_WideResNet')()
        
        if net_name == 'WideResNet_ABC_FT':
            import models.nets.wrn_ABC_FT as net
            builder = getattr(net, 'build_WideResNet')()
        
        else:
            assert Exception("Not Implemented Error")
            
        setattr_cls_from_kwargs(builder, net_conf)
        return builder.build

    
def test_net_builder(net_name, from_name, net_conf=None):
    builder = net_builder(net_name, from_name, net_conf)
    print(f"net_name: {net_name}, from_name: {from_name}, net_conf: {net_conf}")
    print(builder)

    
def get_logger(name, save_path=None, level='INFO'):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    
    log_format = logging.Formatter('[%(asctime)s %(levelname)s] %(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(log_format)
    logger.addHandler(streamHandler)
    
    if not save_path is None:
        os.makedirs(save_path, exist_ok=True)
        fileHandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)
    
    return logger


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix, roc_auc_score, precision_recall_curve
from imblearn.metrics import geometric_mean_score
from scipy import stats
from scipy.stats.mstats import gmean


import numpy as np
from sklearn.metrics import roc_auc_score

def multiclass_auc_roc(y_true, y_prob):
    """
    Calculate the AUC-ROC score for a multiclass classification problem.
    
    Parameters:
    y_true (numpy.ndarray): True labels of shape (N,)
    y_prob (numpy.ndarray): Predicted probabilities of shape (N, K), where K is the number of classes.
    
    Returns:
    float: AUC-ROC score
    """
    auc_roc_scores = []
    
    for class_idx in range(y_prob.shape[1]):
        class_probs = y_prob[:, class_idx]
        class_labels = (y_true == class_idx).astype(int)
        auc_roc = roc_auc_score(class_labels, class_probs)
        auc_roc_scores.append(auc_roc)
    # print(auc_roc_scores)
    mean_auc_roc = np.mean(auc_roc_scores)
    return mean_auc_roc

from sklearn.preprocessing import OneHotEncoder

def auc_m(y_true, y_pred, label1=None, label2=None, freq = False):
    y_true=np.reshape(y_true,[-1,1])
    enc1 = OneHotEncoder()
    enc1.fit(y_true)
    y_true = enc1.transform(y_true).toarray()
    y_pred_shape=np.shape(y_pred)
    if len(y_pred_shape)==1 or y_pred_shape[1]==1:
        y_pred = np.reshape(y_pred, [-1, 1])
        y_pred = enc1.transform(y_pred).toarray()
    def auc_binary(i, j):
        msk = np.logical_or(y_true.argmax(axis=1) == i, y_true.argmax(axis=1) == j)
        return roc_auc_score(y_true[:, i][msk], y_pred[:, i][msk])
    n = y_true.shape[1]
    
    if not freq:
        return np.mean([auc_binary(i, j) for i in range(n) for j in range(n) if i != j])
    else:
        return auc_binary(label1, label2)

def get_metrics(outputs, labels, classes, tag="test/", probs=None):
    '''
    returns a dictionary of computed metrics
    ARGS
        outputs: (np.ndarray) a (N, # classes) dimensional array of output logits of the model
        labels: (np.ndarray) a (N) dimensional array where each element is the ground truth
                index of the corresponding output element
        classes: (list) a list of stings of names of classes
    RETURNS:
        a dictionary of classification metircs, support for:
        1. precision,
        2. recall,
        3. accuracy,
        4. max precision across all classes
        5. mean precision across all classes
        6. min precision  across all classes
        7. max recall  across all classes
        8. mean recall across all classes
        9. min recall  across all classes
        10. f1 micro average
        11. f1 macroa average
        12. Head recall
        13. Tail recall
        14. Head Coverage
        15. Tail Coverage
    '''
    num_classes = len(classes)
    precision = precision_score(labels, outputs, average=None, zero_division=0)
    precision_avg = precision_score(labels, outputs, average='macro', zero_division=0)
    max_precision = np.max(precision)
    min_precision = np.min(precision)
    mean_precision = np.mean(precision)

    recall = recall_score(labels, outputs, average=None, zero_division=0)
    tail_recall = np.mean(recall[int(0.9*num_classes):])
    head_recall = np.mean(recall[:int(0.9*num_classes)])

    minHT = min(tail_recall, head_recall)

    recall_avg = recall_score(labels, outputs, average='macro', zero_division=0)
    max_recall = np.max(recall)
    min_recall = np.min(recall)
    mean_recall = np.mean(recall)

    f1_micro = f1_score(labels, outputs, average='micro')
    f1_macro = f1_score(labels, outputs, average='macro')

    Gmean = gmean(recall)
    Hmean = stats.hmean(recall)

    CM = confusion_matrix(labels, outputs, normalize="all")
    coverages = np.sum(CM, axis=0)

    head_coverage, tail_coverage =  np.mean(coverages[:int(0.9*num_classes)]), \
                                    np.mean(coverages[int(0.9*num_classes):])
    accuracy = accuracy_score(labels, outputs)
    if probs is None:
        aucroc = 0.0
    else:
        aucroc = auc_m(labels, outputs, None, None, False)
        #multiclass_auc_roc(labels, probs)
        #roc_auc_score(labels, probs, multi_class='ovr', max_fpr=1.0, average=None)
        # prcurve = precision_recall_curve(labels, probs)
        # roc_auc_score(labels, probs, multi_class='ovo', max_fpr=None)

    print("aucroc", aucroc)
    # print("prcurve", prcurve)
    metrics =   {
                "precision": precision_avg,
                "recall": recall_avg,
                "accuracy": accuracy,
                "max_precision": max_precision,
                "mean_precision": mean_precision,
                "min_precision": min_precision,
                "max_recall": max_recall,
                "mean_recall": mean_recall,
                "min_recall": min_recall,
                "f1_micro": f1_micro,
                "f1_macro": f1_macro,
                "tail_recall": tail_recall,
                "head_recall": head_recall,
                "min_head_tail": minHT,
                "head_coverage": head_coverage,
                "tail_coverage": tail_coverage,
                "G-mean": Gmean,
                "H-mean": Hmean
                }

    for i, name in enumerate(classes):
        metrics["precision_" + name] = precision[i]
        metrics["recall_" + name] = recall[i]

    for i, name in enumerate(classes):
        metrics["coverage_" + name] = coverages[i]

    metrics["min_coverage"] = min(coverages)

    # mutate the keys to add the tag:
    curr_keys = list(metrics.keys())
    for key in curr_keys:
        metrics[tag + key] = metrics[key]
        del metrics[key]
    print(metrics)
    return metrics, CM


import numpy as np
import torch
import torch.nn.functional as F


def joint_distribution(CM, lambdas, log_derivative=False, T=1.0):
    num_classes = CM.shape[0]
    CM = CM/np.sum(CM)  # normalising the distribution
    CM_row_sum = np.sum(CM, axis=1)  # a commonly used term in the derivative 
    derivative_matrix = np.zeros((num_classes, num_classes))

    for i in range(num_classes):
        for j in range(num_classes):
            if j==i:
                derivative_matrix[i,j] = (-1 * lambdas[i]/CM_row_sum[i]) + (-1 * lambdas[i] * CM[i,i]/(CM_row_sum[i]**2))
            else:
                derivative_matrix[i,j] = -1 * lambdas[i]/(CM_row_sum[i]**2)

    negative_derivative_matrix = torch.tensor(-1 * derivative_matrix/T)
    if log_derivative:
        print("Using the log derivative for sampling here")
        return F.softmax(torch.tensor(CM) * negative_derivative_matrix)
    else:
        return F.softmax(negative_derivative_matrix)


def rand_bbox(size, lam):
    H, W = size[0], size[1]

    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix(tensor1, tensor2):
    batch_size = tensor1.shape[0]
    H,W = tensor1.shape[1], tensor1.shape[2]

    for i in range(batch_size):
        lam = np.random.beta(1.0, 1.0)
        bbx1, bby1, bbx2, bby2 = rand_bbox((H, W), lam)
        tensor1[i, bbx1:bbx2, bby1:bby2] = tensor2[i, bbx1:bbx2, bby1:bby2]
    return tensor1
