#import needed library
import os
import yaml
import logging
import random
import warnings
import cgi
import warnings

# Third-Party Library Imports
from models.wrapper import TimmModelWrapper
from models.nets import wrn
import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import timm


# Local Imports
from utils import get_logger, count_parameters
from train_utils import get_SGD, get_cosine_schedule_with_warmup, WarmupMultiStepLR
from pretraining.semisupervised.FixMatch.FixMatch import FixMatch
from datasets.cifar import CIFAR_SSL_LT_Dataset
from datasets.stl import STL_SSL_LT_Dataset
from datasets.imagenet100 import ImageNet100_SSL_LT_Dataset
from datasets.data_utils import get_data_loader
from configs.yaml_object import YAMLObject


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
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args)) 
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

    if args.net in timm.list_models():
        base_net = timm.create_model(args.net, num_classes=args.num_classes)
    if "wide_resnet28_2" in args.net:
        net_builder = wrn.build_WideResNet(depth=args.depth, widen_factor=args.widen_factor,
                                           bn_momentum=args.bn_momentum, leaky_slope=args.leaky_slope,
                                           dropRate=args.dropout)
        base_net = net_builder.build(args.num_classes)
    net = TimmModelWrapper(base_net, 1.0)

    model = FixMatch(net,
                     args.num_classes,
                     args.ema_m,
                     args.T,
                     args.p_cutoff,
                     args.ulb_loss_ratio,
                     args.hard_label,
                     num_eval_iter=args.num_eval_iter)

    logger.info(f'Number of Trainable Params: {count_parameters(model.train_model)}')

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
            model.train_model.cuda(args.gpu)
            model.train_model = torch.nn.parallel.DistributedDataParallel(model.train_model,
                                                                          device_ids=[args.gpu])
            model.eval_model.cuda(args.gpu)

        else:
            # if arg.gpu is None, DDP will divide and allocate batch_size
            # to all available GPUs if device_ids are not set.
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)

    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model.train_model = model.train_model.cuda(args.gpu)
        model.eval_model = model.eval_model.cuda(args.gpu)

    else:
        model.train_model = torch.nn.DataParallel(model.train_model).cuda()
        model.eval_model = torch.nn.DataParallel(model.eval_model).cuda()

    logger.info(f"model_arch: {model}")
    logger.info(f"Arguments: {args}")

    cudnn.benchmark = True

    # Construct Dataset & DataLoader
    
    if 'cifar' in args.dataset:
        dataset = CIFAR_SSL_LT_Dataset(name=args.dataset, num_classes=args.num_classes, data_dir='./data',
                                    N1=args.N1, M1=args.M1, include_train=args.include_train, uratio=args.uratio, 
                                    imbalance_l=args.imbalance_l, imbalance_u=args.imbalance_u, use_strong_transform=True)
        lb_dset, ulb_dset, val_dset, test_dset  = dataset.return_splits()

    elif 'stl' in args.dataset:
        dataset = STL_SSL_LT_Dataset(name=args.dataset, num_classes=args.num_classes, data_dir='./data',\
                                     N1=args.N1, include_train=False, imbalance_l=args.imbalance_l,\
                                     use_strong_transform=True, size=args.size)
        lb_dset, ulb_dset, val_dset, test_dset  = dataset.return_splits()
    
    if 'imagenet100' in args.dataset:
        dataset = ImageNet100_SSL_LT_Dataset(data_dir=args.data_dir, num_classes=args.num_classes, N1=args.N1, M1=args.M1,\
                                                imbalance_l=args.imbalance_l, imbalance_u=args.imbalance_u, use_strong_transform=True)
        lb_dset, ulb_dset, val_dset, test_dset  = dataset.return_splits()


    loader_dict = {}
    dset_dict = {'train_lb': lb_dset, 'train_ulb': ulb_dset, 'eval': test_dset}

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

    loader_dict['eval'] = get_data_loader(dset_dict['eval'],
                                          args.eval_batch_size, 
                                          num_workers=args.num_workers)

    ## set DataLoader on FixMatch
    model.set_dataset(lb_dset=lb_dset,ulb_dset=ulb_dset,\
                      val_dset=val_dset, test_dset=test_dset,\
                      loader_dict=loader_dict)  # type: ignore

    # SET Optimizer & LR Scheduler
    ## construct SGD and cosine lr scheduler
    optimizer = get_SGD(model.train_model, 'SGD', args.lr, args.momentum, args.weight_decay)
    if "cosine" in args.scheduler:
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    args.num_train_iter,
                                                    num_warmup_steps=args.num_train_iter*0)
    if "multistepLR" in args.scheduler:
        iters_per_epoch = len(dset_dict['train_lb'])//args.batch_size
        milestone_epochs = [60,120, 180, 200]
        milestone_ites = [x * iters_per_epoch for x in milestone_epochs]
        warmup_iters = 5 * iters_per_epoch

        scheduler = WarmupMultiStepLR(optimizer=optimizer, milestones=milestone_ites, gamma=0.1,\
                                      warmup_iters=warmup_iters, warmup_factor=0.001)

    ## set SGD and cosine lr on FixMatch 
    model.set_optimizer(optimizer, scheduler)

    #If args.resume, load checkpoints from args.load_path
    if args.resume:
        model.load_model(args.load_path) #type: ignore

    # START TRAINING of FixMatch
    trainer = model.train
    for epoch in range(args.epoch):
        trainer(args)

    if not args.multiprocessing_distributed or \
                (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        model.save_model('latest_model.pth', save_path) #type: ignore

    logging.warning(f"GPU {args.rank} training is FINISHED")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='')
    warnings.filterwarnings("ignore")
    '''
    Saving & loading of the model.
    '''
    parser.add_argument('--config_file', type=str, default='./configs/sample.yaml')
    args = parser.parse_args()
    with open(args.config_file, "r") as file:
        data = yaml.safe_load(file)

    # Convert dictionary to object
    yaml_object = YAMLObject(**data)
    
    
    main(yaml_object)
