import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.cuda.amp import autocast, GradScaler

import os
import contextlib
from train_utils import AverageMeter

from .fixmatch_utils import consistency_loss, Get_Scalar
from train_utils import ce_loss
from utils import get_metrics
from dataloaders.JointSampling import JointSampler
from MetricOptimisation import JointDistributions, LambdaUpdate

import numpy as np
from sklearn.metrics import recall_score, confusion_matrix
import wandb



class FixMatch:
    def __init__(self, net_builder, num_classes, ema_m, T, p_cutoff, lambda_u,\
                 hard_label=True, t_fn=None, p_fn=None, it=0, num_eval_iter=1000,\
                 tb_log=None, logger=None, classes=None, lb_dataset=None, ulb_dataset=None, args=None):
        """
        class Fixmatch contains setter of data_loader, optimizer, and model update methods.
        Args:
            net_builder: backbone network class (see net_builder in utils.py)
            num_classes: # of label classes 
            ema_m: momentum of exponential moving average for eval_model
            T: Temperature scaling parameter for output sharpening (only when hard_label = False)
            p_cutoff: confidence cutoff parameters for loss masking
            lambda_u: ratio of unsupervised loss to supervised loss
            hard_label: If True, consistency regularization use a hard pseudo label.
            it: initial iteration count
            num_eval_iter: freqeuncy of iteration (after 500,000 iters)
            tb_log: tensorboard writer (see train_utils.py)
            logger: logger (see utils.py)
        """

        super(FixMatch, self).__init__()

        # momentum update param
        self.loader = {}
        self.num_classes = num_classes
        self.ema_m = ema_m
        self.args = args
        # create the encoders
        # network is builded only by num_classes,
        # other configs are covered in main.py
        
        self.train_model = net_builder(num_classes=num_classes) 
        self.eval_model = net_builder(num_classes=num_classes)
        self.num_eval_iter = num_eval_iter
        self.t_fn = Get_Scalar(T) #temperature params function
        self.p_fn = Get_Scalar(p_cutoff) #confidence cutoff function
        self.lambda_u = lambda_u
        self.tb_log = tb_log
        self.use_hard_label = hard_label
        self.num_feats = self.train_model.fc.in_features
        self.optimizer = None
        self.scheduler = None

        self.lb_dataset = lb_dataset
        self.ulb_dataset = ulb_dataset
        
        self.it = 0
        self.classes = None
        self.logger = logger
        self.print_fn = print if logger is None else logger.info
        self.lambdas = [1/self.num_classes] * self.num_classes

        for param_q, param_k in zip(self.train_model.parameters(), self.eval_model.parameters()):
            param_k.data.copy_(param_q.detach().data)  # initialize
            param_k.requires_grad = False  # not update by gradient for eval_net
            
        self.eval_model.eval()


    @torch.no_grad()
    def _eval_model_update(self):
        """
        Momentum update of evaluation model (exponential moving average)
        """
        try:
            for param_train, param_eval in zip(self.train_model.module.parameters(), self.eval_model.parameters()):
                param_eval.copy_(param_eval * self.ema_m + param_train.detach() * (1-self.ema_m))
        except:
            for param_train, param_eval in zip(self.train_model.parameters(), self.eval_model.parameters()):
                param_eval.copy_(param_eval * self.ema_m + param_train.detach() * (1-self.ema_m))
        
        for buffer_train, buffer_eval in zip(self.train_model.buffers(), self.eval_model.buffers()):
            buffer_eval.copy_(buffer_train)            
    
     
    def set_data_loader(self, loader_dict):
        self.loader_dict = loader_dict
        self.print_fn(f'[!] data loader keys: {self.loader_dict.keys()}')    

    
    def set_optimizer(self, optimizer, scheduler=None):
        self.optimizer = optimizer
        self.scheduler = scheduler


    def train(self, args, logger=None):
        """
        Train function of FixMatch.
        From data_loader, it inference training data, computes losses, and update the networks.
        """
        ngpus_per_node = torch.cuda.device_count()
        if not args.multiprocessing_distributed or \
                    (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
            wandb.init(project=args.wandb_project, id=args.wandb_runid, entity=args.wandb_entity)

        #lb: labeled, ulb: unlabeled
        self.train_model.train()

        # for gpu profiling
        start_batch = torch.cuda.Event(enable_timing=True)
        end_batch = torch.cuda.Event(enable_timing=True)
        start_run = torch.cuda.Event(enable_timing=True)
        end_run = torch.cuda.Event(enable_timing=True)
        
        start_batch.record()
        best_eval_acc, best_it = 0.0, 0

        scaler = GradScaler()
        amp_cm = autocast if args.amp else contextlib.nullcontext

        eval_dict, test_metrics = self.evaluate(args=args)
        val_dict, val_metrics = self.val(args=args)

        print(" ===== Data Loaders Have been initialised ======")
        print(" ====== Begining training ======")

        while self.it < args.num_train_iter:
            #tensorboard_dict update
            tb_dict = {}
            if self.it % self.num_eval_iter == 0:
                eval_dict, test_metrics = self.evaluate(args=args)
                val_dict, val_metrics = self.val(args=args)
                tb_dict.update(eval_dict)
                save_path = os.path.join(args.save_dir, args.save_name)

                if tb_dict['eval/top-1-acc'] > best_eval_acc:
                    best_eval_acc = tb_dict['eval/top-1-acc']
                    best_it = self.it

                self.print_fn(f"{self.it} iteration, USE_EMA: {hasattr(self, 'eval_model')}, {tb_dict}, BEST_EVAL_ACC: {best_eval_acc}, at {best_it} iters")

            if not args.multiprocessing_distributed or \
                    (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                if self.it % self.num_eval_iter == 0:
                    wandb.log(test_metrics|val_metrics)

                if self.it == best_it:
                    self.save_model('model_best.pth', save_path)

                if self.it % self.num_eval_iter == 0:
                    save_path = os.path.join(args.save_dir, args.save_name )
                    self.save_model('model_' + "iter:_" + str(self.it) + '_.pth', save_path)

                if not self.tb_log is None:
                    self.tb_log.update(tb_dict, self.it)
            self.train_model.train()
            end_batch.record()
            torch.cuda.synchronize()
            start_run.record()

            x_lb_MO, y_lb_MO, u_w_MO, u_s_MO, y_pl_MO = self.loader_dict['MixupSampler'].get_batch()
            x_lb_MO, y_lb_MO, u_w_MO, u_s_MO, y_pl_MO = x_lb_MO.cuda(args.gpu),\
                                                        y_lb_MO.cuda(args.gpu),\
                                                        u_w_MO.cuda(args.gpu),\
                                                        u_s_MO.cuda(args.gpu),\
                                                        y_pl_MO.cuda(args.gpu)

            num_lb = x_lb_MO.shape[0]
            num_ulb = u_w_MO.shape[0]
            assert num_ulb == num_lb

            with amp_cm():
                logits = self.train_model(x_lb_MO, u_w_MO, only_classify=True)
                total_loss = F.cross_entropy(logits, y_lb_MO, reduction="mean")

            # parameter updates
            if args.amp:
                scaler.scale(total_loss).backward()
                # print(self.train_model.module.conv1.weight.grad)
                # print(self.train_model.module.fc.weight.grad)
                # print(self.train_model.module.bn1.weight.grad)
                scaler.step(self.optimizer)
                scaler.update()
            else:
                total_loss.backward()
                self.optimizer.step()

            self.scheduler.step()
            self.train_model.zero_grad()

            with torch.no_grad():
                self._eval_model_update()

            end_run.record()
            torch.cuda.synchronize()


            tb_dict['train/sup_loss'] = total_loss.detach() 
            tb_dict['train/total_loss'] = total_loss.detach() 
            # tb_dict['train/mask_ratio'] = 1.0 - mask.detach() 
            tb_dict['lr'] = self.optimizer.param_groups[0]['lr']
            tb_dict['train/prefecth_time'] = start_batch.elapsed_time(end_batch)/1000.
            tb_dict['train/run_time'] = start_run.elapsed_time(end_run)/1000.

                
            self.it +=1
            del tb_dict
            start_batch.record()

        eval_dict, test_metrics = self.evaluate(args=args)
        eval_dict.update({'eval/best_acc': best_eval_acc, 'eval/best_it': best_it})
        return eval_dict


    @torch.no_grad()
    def evaluate(self, eval_loader=None, args=None):
        use_ema = hasattr(self, 'eval_model')

        eval_model = self.eval_model if use_ema else self.train_model
        eval_model.eval()
        if eval_loader is None:
            eval_loader = self.loader_dict['eval']

        total_loss = 0.0
        total_acc = 0.0
        total_num = 0.0
        logits_, labels_ = [], []
        for x, y in eval_loader:
            x, y = x.cuda(args.gpu), y.cuda(args.gpu)
            num_batch = x.shape[0]
            total_num += num_batch
            logits = eval_model(x)

            logits_.append(torch.argmax(logits, dim=1).cpu().detach().numpy())
            labels_.append(y.cpu().detach().numpy())

            loss = F.cross_entropy(logits, y, reduction='mean')
            acc = torch.sum(torch.max(logits, dim=-1)[1] == y)

            total_loss += loss.detach()*num_batch
            total_acc += acc.detach()

        if not use_ema:
            eval_model.train()
        logits_, labels_ = (np.concatenate(logits_, axis=0), np.concatenate(labels_, axis=0))
        metrics, _ = get_metrics(logits_, labels_, self.classes, tag="test/")
        return {'eval/loss': total_loss/total_num, 'eval/top-1-acc': total_acc/total_num}, metrics


    @torch.no_grad()
    def val(self, args=None):
        use_ema = hasattr(self, 'eval_model')

        eval_model = self.eval_model if use_ema else self.train_model
        eval_model.eval()
        eval_loader = self.loader_dict['val']

        total_loss = 0.0
        total_acc = 0.0
        total_num = 0.0
        logits_, labels_ = [], []
        for x, y in eval_loader:
            x, y = x.cuda(args.gpu), y.cuda(args.gpu)
            num_batch = x.shape[0]
            total_num += num_batch
            logits = eval_model(x)

            logits_.append(torch.argmax(logits, dim=1).cpu().detach().numpy())
            labels_.append(y.cpu().detach().numpy())

            loss = F.cross_entropy(logits, y, reduction='mean')
            acc = torch.sum(torch.max(logits, dim=-1)[1] == y)

            total_loss += loss.detach()*num_batch
            total_acc += acc.detach()

        if not use_ema:
            eval_model.train()
        
        logits_, labels_ = (np.concatenate(logits_, axis=0), np.concatenate(labels_, axis=0))
        val_metrics, CM = get_metrics(logits_, labels_, self.classes, tag="val/")

        self.lambdas = LambdaUpdate.MinRecall(CM, self.lambdas, args.val_lr, type=self.args.update_type, beta=args.beta)
        lambda_dict = {}
        for i in range(self.num_classes):
            lambda_dict["lambda:"+ str(i)] = self.lambdas[i]
        
        print(self.lambdas)
        joint_dist = JointDistributions.MinRecall(CM, self.lambdas, T=args.DistTemp, beta1=args.beta1,\
                                                  beta2=args.beta2, type=args.gain_type)
        JS = JointSampler(self.lb_dataset, self.ulb_dataset, eval_model,\
                          joint_dist, self.num_classes, batch_size=128)
        self.loader_dict["MixupSampler"] = JS
        return {'eval/loss': total_loss/total_num, 'eval/top-1-acc': total_acc/total_num}, val_metrics|lambda_dict


    def save_model(self, save_name, save_path):
        save_filename = os.path.join(save_path, save_name)
        train_model = self.train_model.module if hasattr(self.train_model, 'module') else self.train_model
        eval_model = self.eval_model.module if hasattr(self.eval_model, 'module') else self.eval_model
        torch.save({'train_model': train_model.state_dict(),
                    'eval_model': eval_model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'it': self.it}, save_filename)
        
        self.print_fn(f"model saved: {save_filename}")
    
    
    def load_model(self, load_path):
        checkpoint = torch.load(load_path)
        
        train_model = self.train_model.module if hasattr(self.train_model, 'module') else self.train_model
        eval_model = self.eval_model.module if hasattr(self.eval_model, 'module') else self.eval_model
        
        for key in checkpoint.keys():
            if hasattr(self, key) and getattr(self, key) is not None:
                if 'train_model' in key:
                    train_model.load_state_dict(checkpoint[key])
                elif 'eval_model' in key:
                    eval_model.load_state_dict(checkpoint[key])
                elif key == 'it':
                    self.it = checkpoint[key]
                elif key == 'scheduler':
                    self.scheduler.load_state_dict(checkpoint[key])
                elif key == 'optimizer':
                    # self.optimizer.load_state_dict(checkpoint[key]) 
                    pass
                else:
                    getattr(self, key).load_state_dict(checkpoint[key])
                self.print_fn(f"Check Point Loading: {key} is LOADED")
            else:
                self.print_fn(f"Check Point Loading: {key} is **NOT** LOADED")

if __name__ == "__main__":
    pass
