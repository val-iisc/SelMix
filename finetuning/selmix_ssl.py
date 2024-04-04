import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import os
import contextlib
import numpy as np
import wandb
from utils import get_metrics
from dataloaders import FastJointSampler
from MetricOptimisation import *

class SelMixSSL:
    """
    SelMixSSL class for semi-supervised learning with mixup and selective mixing.

    Args:
        model: Base model for SSL.
        num_classes: Number of label classes.
        num_eval_iter: Frequency of iteration for evaluation.
        args: Command-line arguments.
    """
    def __init__(self, model, args=None):
        super(SelMixSSL, self).__init__()
        self.loader = {}
        self.num_classes = args.num_classes
        self.args = args
        self.model = model
        self.num_eval_iter = args.num_eval_iter
        self.save_after = 10 * args.num_eval_iter
        self.num_feats = self.model.model.fc.in_features
        self.optimizer = None
        self.scheduler = None
        self.it = 0
        self.classes = [str(i) for i in range(self.num_classes)]
        self.prior = None
        
        self.MaxGain = [1000]
        self.temperature = float(args.DistTemp)
        self.objective_name = str(args.M)

        self.beta = float(args.beta)
        self.alpha = float(args.alpha)
        self.tau = float(args.tau)
        self.val_lr = float(args.val_lr)

        if "coverage" in self.objective_name:
            self.lambdas = [0] * self.num_classes
        else:
            self.lambdas = [1 / self.num_classes] * self.num_classes

        self.model.eval()

    def set_data_loader(self, loader_dict):
        self.loader_dict = loader_dict
        print(f'[!] Data loader keys: {self.loader_dict.keys()}')

    def set_optimizer(self, optimizer, scheduler=None):
        self.optimizer = optimizer
        self.scheduler = scheduler
    
    def set_dataset(self, lb_dset, ulb_dset, val_dset, test_dset, loader_dict):
        self.lb_dataset = lb_dset
        self.ulb_dataset = ulb_dset
        self.val_dataset = val_dset
        self.test_dataset = test_dset
        self.loader_dict = loader_dict
        self.prior = lb_dset.prior
        print(f'[!] data loader keys: {self.loader_dict.keys()}')

    def train(self, args):
        ngpus_per_node = torch.cuda.device_count()

        def should_log():
            return not args.multiprocessing_distributed or \
                   (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0)

        def log_to_wandb(metrics):
            if should_log():
                lambda_dict = {f"lambda:{i}": self.lambdas[i] for i in range(self.num_classes)}
                wandb.log(metrics | lambda_dict)

        def save_best_model():
            nonlocal best_eval_acc, best_it
            if self.it == best_it:
                self.save_model('model_best.pth', os.path.join(args.save_dir, args.save_name))

        def save_iter_model():
            if self.it % self.num_eval_iter == 0:
                save_path = os.path.join(args.save_dir, args.save_name)
                self.save_model(f'model_iter_{self.it}.pth', save_path)

        if should_log():
            wandb.init(project=args.wandb_project, id=args.wandb_runid, entity=args.wandb_entity)

        self.model.train()

        start_batch = torch.cuda.Event(enable_timing=True)
        end_batch = torch.cuda.Event(enable_timing=True)
        start_run = torch.cuda.Event(enable_timing=True)
        end_run = torch.cuda.Event(enable_timing=True)

        start_batch.record()
        best_eval_acc, best_it = 0.0, 0

        scaler = GradScaler()
        amp_cm = autocast if args.amp else contextlib.nullcontext

        test_metrics = self.evaluate()
        val_metrics = self.val(args=args)

        print(" ===== Data Loaders Have been initialized ======")
        print(" ====== Beginning training ======")
        self.it = 0

        while self.it < args.num_train_iter:
            self.model.train()
            for module in self.model.model.modules():
                if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
                    # print("changing")
                    module.momentum = args.bn_momentum
                    module.track_running_stats = False
                    module.requires_grad_ = False
            
            if self.it % 50 == 0:
                test_metrics = self.evaluate()
                val_metrics = self.val(args=args)
                save_best_model()
                save_iter_model()

            if should_log():
                log_to_wandb(test_metrics | val_metrics)

            end_batch.record()
            torch.cuda.synchronize()
            start_run.record()

            x_lb_MO, y_lb_MO, u_w_MO, y_pl_MO = self.loader_dict['MixupSampler'].get_batch()
            x_lb_MO, y_lb_MO, u_w_MO, y_pl_MO = x_lb_MO.cuda(args.gpu), \
                                               y_lb_MO.cuda(args.gpu), \
                                               u_w_MO.cuda(args.gpu), \
                                               y_pl_MO.cuda(args.gpu)

            num_lb = x_lb_MO.shape[0]
            num_ulb = u_w_MO.shape[0]
            assert num_ulb == num_lb

            with amp_cm():
                logits = self.model(x_lb_MO, u_w_MO)
                total_loss = F.cross_entropy(logits / self.args.T, y_lb_MO, reduction="mean")

            if args.amp:
                scaler.scale(total_loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
            else:
                total_loss.backward()
                self.optimizer.step()

            self.scheduler.step()
            self.model.zero_grad()
            self.optimizer.zero_grad()

            end_run.record()
            torch.cuda.synchronize()

            self.it += 1
            start_batch.record()

        test_metrics = self.evaluate()
        return

    @torch.no_grad()
    def feedforward(self, dataloader, return_feats=True):
        self.model.eval()

        preds, labels, features = [], [], []

        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.cuda(self.args.gpu), y.cuda(self.args.gpu)

                out = self.model(x)

                feats = self.model.pre_logits(x)
                feats = feats.cpu().detach().tolist()
                features.extend(feats)

                preds.append(torch.argmax(out, dim=1).cpu().detach().numpy())
                labels.append(y.cpu().detach().numpy())

            preds, labels = (np.concatenate(preds, axis=0), np.concatenate(labels, axis=0))
        return labels, preds, features

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        eval_loader = self.loader_dict['eval']

        labels, preds, _ = self.feedforward(eval_loader)
        metrics, _ = get_metrics(preds, labels, self.classes, tag="test/")
        print("Acc:", self.it, "is : ", metrics["test/mean_recall"])
        print("Acc:", self.it, "is : ", metrics["test/min_coverage"])
        
        return metrics

    @torch.no_grad()
    def val(self, args=None):
        self.model.eval()

        val_loader = self.loader_dict['val']

        model = self.model
        model.eval()
        labels, preds, features = self.feedforward(val_loader, return_feats=True)

        prototypes = np.zeros((self.num_classes, self.num_feats))
        features = np.array(features)
        for class_label in range(self.num_classes):
            feats_for_class = features[labels == class_label]
            prototypes[class_label] = np.mean(feats_for_class, axis=0)

        val_metrics, CM = get_metrics(preds, labels, self.classes, tag="val/")

        if self.objective_name == "mean_recall":
            objective = MeanRecall(CM, prototypes, model, self.temperature)

        elif self.objective_name == "min_recall":
            objective = MinRecall(CM, prototypes, model, self.temperature,\
                           self.lambdas, self.beta, self.val_lr)
            self.lambdas = objective.lambdas

        elif self.objective_name == "min_HT_recall":
            objective = MinHTRecall(CM, prototypes, model, self.temperature,\
                           self.lambdas, self.beta, self.val_lr)
            self.lambdas = objective.lambdas

        elif self.objective_name == "mean_recall_min_coverage":
            objective = MeanRecallWithCoverage(CM, prototypes, model,\
                                        self.temperature, self.lambdas,\
                                        alpha=self.alpha, tau=self.tau,
                                        lambda_max=self.args.lambda_max)
            self.lambdas = objective.lambdas
        elif self.objective_name == "mean_recall_min_HT_coverage":
            objective = MeanRecallWithHTCoverage(CM, prototypes, model,\
                                          self.temperature, self.lambdas,
                                          alpha=self.alpha, tau=self.tau,
                                          lambda_max=self.args.lambda_max)
            self.lambdas = objective.lambdas
        elif self.objective_name == "HM_min_HT_coverage":
            objective = HmeanWithHTCoverage(CM, prototypes, self.train_model,\
                                     self.temperature, self.lambdas,\
                                     alpha=self.alpha, tau=self.tau,
                                     lambda_max=self.args.lambda_max)
            self.lambdas = objective.lambdas
        elif self.objective_name == "g_mean":
            objective = Gmean(CM, prototypes, model, self.temperature)

        elif self.objective_name == "h_mean":
            objective = Hmean(CM, prototypes, model, self.temperature)

        elif self.objective_name == "HM_min_coverage":
            objective = HmeanWithCoverage(CM, prototypes, model,\
                                self.temperature,\
                                alpha=self.alpha, tau=self.tau,
                                lambda_max=self.args.lambda_max)
            self.lambdas = objective.lambdas

        self.P = objective.P
        self.MaxGain.append(np.max(objective.G))

        if self.has_converged():
            self.it = self.args.num_train_iter -1

        JS = FastJointSampler(self.lb_dataset, self.ulb_dataset, model,
                              samp_dist=self.P, batch_size=args.batch_size, 
                              semi_supervised=False)

        self.objective = objective
        self.loader_dict["MixupSampler"] = JS
        print(" val Acc:", self.it, "is : ", val_metrics["val/mean_recall"])
        return val_metrics

    def save_model(self, save_name, save_path):
        save_filename = os.path.join(save_path, save_name)
        self.model = self.model.module if hasattr(self.model, 'module') else self.model
        torch.save({'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'it': self.it}, save_filename)

        print(f"Model saved: {save_filename}")

    def has_converged(self, window_size=10, tolerance_percentage=0.05):
        if len(self.MaxGain) < window_size:
            return False
        window = self.MaxGain[-window_size:]
        mean = sum(window) / window_size
        return abs(mean - self.MaxGain[-1]) < tolerance_percentage * abs(mean)

    def load_model(self, load_path):
        checkpoint = torch.load(load_path)
        print(checkpoint.keys())
        self.model = self.model.module if hasattr(self.model, 'module') else self.model
        self.model.load_state_dict(checkpoint['eval_model'])
        for key in checkpoint.keys():
            if hasattr(self, key) and getattr(self, key) is not None:
                if 'eval_model' in key:
                    self.model.load_state_dict(checkpoint[key])
                    print("checkpint loaded")
                elif key == 'it':
                    self.it = checkpoint[key]
                print(f"Checkpoint Loading: {key} is LOADED")
            else:
                print(f"Checkpoint Loading: {key} is **NOT** LOADED")


if __name__ == "__main__":
    pass