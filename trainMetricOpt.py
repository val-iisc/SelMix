#import needed library
import os
import logging
import random
import warnings

import wandb
import timm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp

from models.wrapper import TimmModelWrapper
from utils import net_builder, get_logger, count_parameters
from train_utils import TBLog, get_finetune_SGD
from finetuning.selmix_ssl import SelMixSSL
from datasets.cifar import CIFAR_SSL_LT_Dataset
from datasets.data_utils import get_data_loader



def main(args):
    '''
    For (Distributed)DataParallelism,
    main(args) spawn each process (main_worker) to each GPU.
    '''
    save_path = os.path.join(args.save_dir, args.save_name)
    if os.path.exists(save_path) and not args.overwrite:
        raise Exception('already existing model: {}'.format(save_path))
    if args.resume:
        if args.load_path is None:
            raise Exception('Resume of training requires --load_path in the args')
        if os.path.abspath(save_path) == os.path.abspath(args.load_path) and not args.overwrite:
            raise Exception('Saving & Loading pathes are same. \
                            If you want over-write, give --overwrite in the argument.')

    if args.seed is not None:
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    #distributed: true if manually selected or if world_size > 1
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed 
    ngpus_per_node = torch.cuda.device_count() # number of gpus of each node

    #divide the batch_size according to the number of nodes
    args.batch_size = int(args.batch_size / args.world_size)

    if args.multiprocessing_distributed:
        # now, args.world_size means num of total processes in all nodes
        args.world_size = ngpus_per_node * args.world_size 

        #args=(,) means the arguments of main_worker
        torch.multiprocessing.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args)) 
    else:
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    '''
    main_worker is conducted on each GPU.
    '''

    global best_acc1
    args.gpu = gpu

    # random seed has to be set for the syncronization of labeled data sampling in each process.
    assert args.seed is not None
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True


    # SET UP FOR DISTRIBUTED TRAINING
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu # compute global rank

        # set distributed group:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)


    #SET save_path and logger
    save_path = os.path.join(args.save_dir, args.save_name)
    logger_level = "WARNING"
    if args.rank % ngpus_per_node == 0:
        logger_level = "INFO"

    logger = get_logger(args.save_name, save_path, logger_level)
    logger.warning(f"USE GPU: {args.gpu} for training")
    
    net_timm = timm.create_model(args.net, num_classes=args.num_classes)
    net = TimmModelWrapper(net_timm, 0.6)
    for module in net.model.modules():
        if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
            module.momentum = 0.0
            module.track_running_stats = False
            module.requires_grad_ = False
    if 'bn' in [name for name, _ in net_timm.named_modules()]:
        # Set the Batch Normalization momentum
        # Freezing bn update to preserve the 
        # condition of fixed prototype assumption
        bn_momentum = 0.0  
        for module in net_timm.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.momentum = bn_momentum
                module.requires_grad_ = False

    # SET FixMatch: class FixMatch in models.fixmatch
    model = SelMixSSL(net,
                      args, 
                      args=args)

    logger.info(f'Number of Trainable Params: {count_parameters(model.model)}')

    # SET Optimizer & LR Scheduler
    ## construct SGD and cosine lr scheduler

    optimizer = get_finetune_SGD(model.model, args.opt,\
                    lr = args.lr, weight_decay=args.weight_decay,\
                    freeze_backbone=args.freeze_backbone)

    for name,param in model.model.named_parameters():
        if param.requires_grad:
            print(name, param.requires_grad)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_train_iter,\
                                                           eta_min=0, last_epoch=- 1, verbose=False)
    ## set SGD and cosine lr on FixMatch 
    model.set_optimizer(optimizer, scheduler)


    # SET Devices for (Distributed) DataParallel
    if not torch.cuda.is_available():
        raise Exception('ONLY GPU TRAINING IS SUPPORTED')
    elif args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            '''
            batch_size: batch_size per node -> batch_size per gpu
            workers: workers per node -> workers per gpu
            '''
            args.batch_size = int(args.batch_size / ngpus_per_node)
            model.model.cuda(args.gpu)
            model.model = torch.nn.parallel.DistributedDataParallel(model.model,
                                                                          device_ids=[args.gpu], find_unused_parameters=True)
            model.model.cuda(args.gpu)

        else:
            # if arg.gpu is None, DDP will divide and allocate batch_size
            # to all available GPUs if device_ids are not set.
            model.cuda()  
            model = torch.nn.parallel.DistributedDataParallel(model)

    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model.model = model.model.cuda(args.gpu)

    else:
        model.model = torch.nn.DataParallel(model.model).cuda()

    logger.info(f"model_arch: {model}")
    logger.info(f"Arguments: {args}")

    cudnn.benchmark = True

    # Construct Dataset & DataLoader
    if 'cifar' in args.dataset:
        dataset = CIFAR_SSL_LT_Dataset(name=args.dataset, num_classes=args.num_classes, data_dir='./data',
                                    N1=args.N1, M1=args.M1, include_train=False, uratio=args.uratio, 
                                    imbalance_l=args.imbalance_l, imbalance_u=args.imbalance_u)

        lb_dset, ulb_dset, val_dset, test_dset  = dataset.return_splits()

        # add some extra params that are needed post-hoc
        


    elif 'stl' in args.dataset:
        dataset = STL_SSL_LT_Dataset("stl10", 10, args.data_dir, args.N1, False, args.imbalance_l, True, size=args.size)
        lb_dset, ulb_dset, val_dset, test_dset  = dataset.return_splits()
        model.classes = lb_dset.classes
        model.lb_dataset = lb_dset
        model.ulb_dataset = ulb_dset
        model.prior = lb_dset.prior


    loader_dict = {}
    dset_dict = {'train_lb': lb_dset, 'train_ulb': ulb_dset, 'eval': test_dset, 'val': val_dset}

    loader_dict['train_lb'] = get_data_loader(dset_dict['train_lb'],
                                              args.batch_size,
                                              data_sampler = args.train_sampler,
                                              num_iters=args.num_train_iter,
                                              num_workers=args.num_workers, 
                                              distributed=args.distributed)

    loader_dict['train_ulb'] = get_data_loader(dset_dict['train_ulb'],
                                               args.batch_size*args.uratio,
                                               data_sampler = args.train_sampler,
                                               num_iters=args.num_train_iter,
                                               num_workers=4*args.num_workers,
                                               distributed=args.distributed)

    loader_dict['val'] = get_data_loader(dset_dict['val'],
                                          args.eval_batch_size, 
                                          num_workers=args.num_workers)

    loader_dict['eval'] = get_data_loader(dset_dict['eval'],
                                          args.eval_batch_size, 
                                          num_workers=args.num_workers)

    ## set DataLoader on FixMatch
    model.set_dataset(lb_dset=lb_dset,ulb_dset=ulb_dset,\
                      val_dset=val_dset, test_dset=test_dset,\
                      loader_dict=loader_dict)  # type: ignore

    #If args.resume, load checkpoints from args.load_path
    if args.resume:
        model.load_model(args.load_path)

    # START TRAINING of FixMatch
    trainer = model.train
    for epoch in range(args.epoch):
        trainer(args)

    if not args.multiprocessing_distributed or \
                (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        model.save_model('latest_model.pth', save_path)

    logging.warning(f"GPU {args.rank} training is FINISHED")



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='')
    
    '''
    Saving & loading of the model.
    '''
    parser.add_argument('--save_dir', type=str, default='./saved_models')
    parser.add_argument('--save_name', type=str, default='fixmatch')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--nestrov', action='store_true')
    parser.add_argument('--freeze_backbone', action='store_true')
    
    '''
    Training Configuration of FixMatch
    '''
    parser.add_argument('--M', default="mean_recall", type=str)
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--num_train_iter', type=int, default=2**20, 
                        help='total number of training iterations')
    parser.add_argument('--num_eval_iter', type=int, default=10000,
                        help='evaluation frequency')
    parser.add_argument('--num_labels', type=int, default=4000)
    parser.add_argument('--batch_size', type=int, default=64,
                        help='total number of batch size of labeled data')
    parser.add_argument('--eval_batch_size', type=int, default=1024,
                        help='batch size of evaluation data loader (it does not affect the accuracy)')
    
    parser.add_argument('--DistTemp', type=float, default=1.0)
    parser.add_argument('--percentile', type=float, default=100.0)
    parser.add_argument('--ema_m', type=float, default=0.999, help='ema momentum for model')
    parser.add_argument('--ema_v', type=float, default=1.0, help='ema momentum for model')
    parser.add_argument('--mixup_lambda_min', type=float, default=0.6)
    parser.add_argument('--filter', action='store_true', help='use mixed precision training or not')
    parser.add_argument('--conf_thresh', type=float, default=None)
    parser.add_argument('--uratio', type=int, default=7,
                        help='the ratio of unlabeled data to labeld data in each mini-batch')

    '''
    Optimizer configurations
    '''
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--min_lr', type=float, default=0.00)
    parser.add_argument('--val_lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--damp', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--amp', action='store_true', help='use mixed precision training or not')
    parser.add_argument('--p_ema', type=float, default=1.0)
    parser.add_argument('--T', type=float, default=1.0)
    '''
    Backbone Net Configurations
    '''
    parser.add_argument('--net', type=str, default='WideResNet')
    parser.add_argument('--update_type', type=str, default='EG')
    parser.add_argument('--net_from_name', type=bool, default=False)
    parser.add_argument('--depth', type=int, default=28)
    parser.add_argument('--widen_factor', type=int, default=2)
    parser.add_argument('--leaky_slope', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--min_gain', type=float, default=0.0)
    parser.add_argument('--beta', type=float, default=0.0)
    parser.add_argument('--alpha', type=float, default=0.95)
    parser.add_argument('--tau', type=float, default=0.1)
    parser.add_argument('--lambda_max', type=float, default=100.0)

    '''
    Data Configurations
    '''

    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--train_sampler', type=str, default='RandomSampler')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--imbalance_l', type=float, default=1.0)
    parser.add_argument('--frac', type=float, default=1.0)
    parser.add_argument('--imbalance_u', type=float, default=1.0)
    parser.add_argument('--size', type=int, default=32)
    parser.add_argument('--N1', type=int, default=1)
    parser.add_argument('--M1', type=int, default=1)
    parser.add_argument('--lt', action='store_true')
    parser.add_argument('--train_backbone', action='store_true')
    parser.add_argument('--mask', action='store_true')
    parser.add_argument('--cutmix', action='store_true')
    parser.add_argument('--puzzlemix', action='store_true')
    parser.add_argument('--vmix', action='store_true')
    '''
    multi-GPUs & Distrbitued Training
    '''
    
    ## args for distributed training (from https://github.com/pytorch/examples/blob/master/imagenet/main.py)
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='**node rank** for distributed training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:10001', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')

    '''
    wandb logging
    '''
    parser.add_argument('--wandb-project', default="FixMatch",
                        help='directory to output the result', type=str)
    parser.add_argument('--wandb-entity', default="stablegradients",
                        help='directory to output the result', type=str)
    parser.add_argument('--wandb-runid', default="maxmin_recall", type=str)
    parser.add_argument('--opt', default="SGD", type=str)

    args = parser.parse_args()
    main(args)