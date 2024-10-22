import os
import contextlib
from torch.cuda.amp import autocast, GradScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from matplotlib.rcsetup import validate_sketch
import wandb
import copy
import numpy as np

from train_utils import AverageMeter
from .fixmatch_utils import consistency_loss, Get_Scalar
from train_utils import ce_loss
from utils import get_metrics


class FixMatch:
    def __init__(self, model, num_classes, ema_m, T, p_cutoff, lambda_u, hard_label=True, num_eval_iter=1000):
        """
        Class FixMatch contains setter of data_loader, optimizer, and model update methods.
        
        Args:
            model: Backbone network class (see net_builder in utils.py).
            num_classes: Number of label classes.
            ema_m: Momentum of exponential moving average for eval_model.
            T: Temperature scaling parameter for output sharpening (only when hard_label = False).
            p_cutoff: Confidence cutoff parameters for loss masking.
            lambda_u: Ratio of unsupervised loss to supervised loss.
            hard_label: If True, consistency regularization uses a hard pseudo-label.
            num_eval_iter: Frequency of iteration (after 500,000 iters).
        """
        super(FixMatch, self).__init__()

        # Momentum update param
        self.loader = {}
        self.num_classes = num_classes
        self.ema_m = ema_m

        # Create the encoders
        # Network is built only by num_classes,
        # other configs are covered in main.py
        self.train_model = model
        self.eval_model = copy.deepcopy(model)
        self.num_eval_iter = num_eval_iter
        self.save_after = 50 * num_eval_iter
        self.t_fn = Get_Scalar(T)  # Temperature params function
        self.p_fn = Get_Scalar(p_cutoff)  # Confidence cutoff function
        self.lambda_u = lambda_u
        self.use_hard_label = hard_label

        self.optimizer = None
        self.scheduler = None

        self.it = 0
        self.classes = [str(i) for i in range(self.num_classes)]
        self.prior = None

        for param_q, param_k in zip(self.train_model.parameters(), self.eval_model.parameters()):
            param_k.data.copy_(param_q.detach().data)  # Initialize
            param_k.requires_grad = False  # Not update by gradient for eval_net

        self.eval_model.eval()

    @torch.no_grad()
    def _eval_model_update(self):
        """
        Momentum update of the evaluation model (exponential moving average).
        """
        for param_train, param_eval in zip(self.train_model.module.parameters(), self.eval_model.parameters()):
            param_eval.copy_(param_eval * self.ema_m + param_train.detach() * (1 - self.ema_m))

        for buffer_train, buffer_eval in zip(self.train_model.buffers(), self.eval_model.buffers()):
            buffer_eval.copy_(buffer_train)

    def set_optimizer(self, optimizer, scheduler=None):
        """
        Set optimizer and scheduler for the SSL model.

        Args:
            optimizer: Optimizer for the SSL model.
            scheduler: Learning rate scheduler. Default is None.
        """
        self.optimizer = optimizer
        self.scheduler = scheduler

    def set_dataset(self, lb_dset, ulb_dset, val_dset, test_dset, loader_dict):
        """
        Set datasets and loader dictionary for the model.

        Parameters:
            lb_dset: Labeled dataset.
            ulb_dset: Unlabeled dataset.
            val_dset: Validation dataset.
            test_dset: Test dataset.
            loader_dict: Dictionary containing different data loaders.

        Returns:
            None
        """
        self.lb_dataset = lb_dset
        self.ulb_dataset = ulb_dset
        self.val_dataset = val_dset
        self.test_dataset = test_dset
        self.loader_dict = loader_dict
        self.prior = lb_dset.prior
        print(f'[!] data loader keys: {self.loader_dict.keys()}')

    def train(self, args):
        """
        Train function of FixMatch.
        From data_loader, it infers training data, computes losses, and updates the networks.
        """
        ngpus_per_node = torch.cuda.device_count()
        if not args.multiprocessing_distributed or (
                args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
            wandb.init(project=args.wandb_project, id=args.wandb_runid, entity=args.wandb_entity)

        # lb: labeled, ulb: unlabeled
        self.train_model.train()

        # For GPU profiling
        start_batch = torch.cuda.Event(enable_timing=True)
        end_batch = torch.cuda.Event(enable_timing=True)
        start_run = torch.cuda.Event(enable_timing=True)
        end_run = torch.cuda.Event(enable_timing=True)

        start_batch.record() # type: ignore
        best_eval_acc, best_it = 0.0, 0

        scaler = GradScaler()
        amp_cm = autocast if args.amp else contextlib.nullcontext

        for (x_lb, y_lb), ((x_ulb_w, x_ulb_s), _) in zip(self.loader_dict['train_lb'], self.loader_dict['train_ulb']):

            # Prevent the training iterations exceed args.num_train_iter
            if self.it > args.num_train_iter:
                break

            end_batch.record()  # type: ignore
            torch.cuda.synchronize()
            start_run.record()  # type: ignore

            num_lb = x_lb.shape[0]
            num_ulb = x_ulb_w.shape[0]
            assert num_ulb == x_ulb_s.shape[0]

            x_lb, x_ulb_w, x_ulb_s = x_lb.cuda(args.gpu), x_ulb_w.cuda(args.gpu), x_ulb_s.cuda(args.gpu)
            y_lb = y_lb.cuda(args.gpu)

            inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))

            # Inference and calculate sup/unsup losses
            with amp_cm():
                logits = self.train_model(inputs)
                logits_x_lb = logits[:num_lb]
                logits_x_ulb_w, logits_x_ulb_s = logits[num_lb:].chunk(2)
                del logits

                # Hyper-params for update
                T = self.t_fn(self.it)
                p_cutoff = self.p_fn(self.it)
                if args.LA:
                    logits_x_lb = logits_x_lb + torch.log(torch.tensor(self.prior).cuda(args.gpu))
                else:
                    pass
                sup_loss = ce_loss(logits_x_lb, y_lb, reduction='mean')
                unsup_loss, mask = consistency_loss(logits_x_ulb_w,
                                                    logits_x_ulb_s,
                                                    'ce', T, p_cutoff,
                                                    use_hard_labels=args.hard_label)  # type: ignore

                total_loss = sup_loss + self.lambda_u * unsup_loss

            # Parameter updates
            if args.amp:
                scaler.scale(total_loss).backward()
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

            # Tensorboard_dict update
            tb_dict = {}
            tb_dict['train/sup_loss'] = sup_loss.detach()
            tb_dict['train/unsup_loss'] = unsup_loss.detach()
            tb_dict['train/total_loss'] = total_loss.detach()
            tb_dict['train/mask_ratio'] = 1.0 - mask.detach()
            tb_dict['lr'] = self.optimizer.param_groups[0]['lr']
            tb_dict['train/prefetch_time'] = start_batch.elapsed_time(end_batch) / 1000.
            tb_dict['train/run_time'] = start_run.elapsed_time(end_run) / 1000.

            if self.it % self.num_eval_iter == 0:
                eval_dict, metrics = self.evaluate(args=args)
                tb_dict.update(eval_dict)

                save_path = os.path.join(args.save_dir, args.save_name)

                if tb_dict['eval/top-1-acc'] > best_eval_acc:
                    best_eval_acc = tb_dict['eval/top-1-acc']
                    best_it = self.it

                print(
                    f"{self.it} iteration, USE_EMA: {hasattr(self, 'eval_model')}, {tb_dict}, BEST_EVAL_ACC: {best_eval_acc}, at {best_it} iters")

            if not args.multiprocessing_distributed or (
                    args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                if self.it % self.num_eval_iter == 0:
                    wandb.log(metrics)
                if self.it == best_it:
                    self.save_model('model_best.pth', save_path)

                if self.it % self.save_after == 0:
                    save_path = os.path.join(args.save_dir, args.save_name)
                    self.save_model('model_' + "iter:_" + str(self.it) + '_.pth', save_path)

            self.it += 1
            del tb_dict
            start_batch.record()
            if self.it > 2 ** 19:
                self.num_eval_iter = 1000

        eval_dict, metrics = self.evaluate(args=args)
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

            total_loss += loss.detach() * num_batch
            total_acc += acc.detach()

        if not use_ema:
            eval_model.train()
        logits_, labels_ = (np.concatenate(logits_, axis=0), np.concatenate(labels_, axis=0))
        metrics, CM = get_metrics(logits_, labels_, self.classes)

        return {'eval/loss': total_loss / total_num, 'eval/top-1-acc': total_acc / total_num}, metrics

    def save_model(self, save_name, save_path):
        save_filename = os.path.join(save_path, save_name)
        train_model = self.train_model.module if hasattr(self.train_model, 'module') else self.train_model
        eval_model = self.eval_model.module if hasattr(self.eval_model, 'module') else self.eval_model
        torch.save({'train_model': train_model.state_dict(),
                    'eval_model': eval_model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'it': self.it}, save_filename)

        print(f"Model saved: {save_filename}")

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
                    self.optimizer.load_state_dict(checkpoint[key])
                else:
                    getattr(self, key).load_state_dict(checkpoint[key])
                print(f"Checkpoint Loading: {key} is LOADED")
            else:
                print(f"Checkpoint Loading: {key} is **NOT** LOADED")


if __name__ == "__main__":
    pass
