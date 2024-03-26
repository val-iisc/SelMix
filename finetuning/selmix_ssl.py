import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import os
import contextlib
import numpy as np
import wandb
from train_utils import plot_samp_dist, ce_loss
from utils import get_metrics
from dataloaders import FastJointSampler
from MetricOptimisation import *

class SelMixSSL:
    """
    SelMixSSL class for semi-supervised learning with mixup and selective mixing.

    Args:
        model: Base model for SSL.
        num_classes: Number of label classes.
        ema_m: Momentum of exponential moving average for eval_model.
        hard_label: If True, consistency regularization uses a hard pseudo label.
        num_eval_iter: Frequency of iteration for evaluation.
        dataset: Dataset for SSL.
        args: Command-line arguments.
    """
    def __init__(self, model, num_classes, num_eval_iter=1000,  args=None):
        super(SelMixSSL, self).__init__()

        # Initialization of attributes
        self.loader = {}
        self.num_classes = args.num_classes
        self.args = args
        self.model = model
        self.num_eval_iter = num_eval_iter
        self.save_after = 10 * num_eval_iter
        self.num_feats = self.model.model.fc.in_features
        self.optimizer = None
        self.scheduler = None
        self.it = 0
        self.classes = [str(i) for i in range(self.num_classes)]
        self.prior = None
        
        self.temperature = float(args.DistTemp) #type: ignore
        self.objective_name = str(args.M) #type: ignore

        # hyper parameters for the lagrange multiplier update rate
        self.beta = float(args.beta) #type: ignore
        self.alpha = float(args.alpha) #type: ignore
        self.tau = float(args.tau) #type: ignore
        self.val_lr = float(args.val_lr) #type: ignore

        # Initialize lagrange multipliers based on 
        # optimization type, if samples from 
        # R+ or from K-1 simplex
        if "coverage" in self.objective_name:
            self.lambdas = [0] * self.num_classes
        else:
            self.lambdas = [1 / self.num_classes] * self.num_classes

        # Set the model to evaluation mode
        self.model.eval()


    def set_data_loader(self, loader_dict):
        """
        Set data loader for the SSL model.

        Args:
            loader_dict (dict): Dictionary containing data loaders.
        """
        self.loader_dict = loader_dict
        print(f'[!] Data loader keys: {self.loader_dict.keys()}')

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
        Train function of SelMixSSL.

        Args:
            args: Command-line arguments.
            logger: Logger for training progress. Default is None.
        """
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

        # Initialize WandB logging if required
        if should_log():
            wandb.init(project=args.wandb_project, id=args.wandb_runid, entity=args.wandb_entity) 

        # Set the model to training mode
        self.model.train()

        # Initialize timing events
        start_batch = torch.cuda.Event(enable_timing=True)
        end_batch = torch.cuda.Event(enable_timing=True)
        start_run = torch.cuda.Event(enable_timing=True)
        end_run = torch.cuda.Event(enable_timing=True)

        start_batch.record()  # type: ignore
        best_eval_acc, best_it = 0.0, 0

        scaler = GradScaler()
        amp_cm = autocast if args.amp else contextlib.nullcontext

        test_metrics = self.evaluate(args=args)
        val_metrics = self.val(args=args)

        print(" ===== Data Loaders Have been initialized ======")
        print(" ====== Beginning training ======")
        self.it = 0

        while self.it < args.num_train_iter:
            self.model.train()

            if self.it % 5 == 0:
                test_metrics = self.evaluate(args=args)
                val_metrics = self.val(args=args)
                save_best_model()
                save_iter_model()

            if should_log():
                log_to_wandb(test_metrics | val_metrics)

            end_batch.record()  # type: ignore
            torch.cuda.synchronize()
            start_run.record()  # type: ignore

            self.model.train()
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

            # Parameter updates
            if args.amp:
                scaler.scale(total_loss).backward()  # type: ignore
                scaler.step(self.optimizer)
                scaler.update()
            else:
                total_loss.backward()
                self.optimizer.step()

            self.scheduler.step()
            self.model.zero_grad()
            self.optimizer.zero_grad()

            end_run.record()  # type: ignore
            torch.cuda.synchronize()

            self.it += 1
            start_batch.record()  # type: ignore

        test_metrics = self.evaluate(args=args)
        return

    @torch.no_grad()
    def feedforward(self, dataloader, return_feats=True):
        """
        Perform a forward pass through the model on the given dataloader.

        Args:
            dataloader (torch.utils.data.DataLoader): DataLoader containing input data.
            return_feats (bool): Whether to return intermediate features. Default is True.

        Returns:
            tuple: If return_feats is True, returns a tuple (labels, predictions, features).
                   If return_feats is False, returns a tuple (labels, predictions).
        """
        # Set the model to evaluation mode
        self.model.eval()

        # Lists to store predictions, labels, and features
        preds, labels, features = [], [], []

        # Disable gradient calculation during inference
        with torch.no_grad():
            for x, y in dataloader:
                # Move data to GPU
                x, y = x.cuda(self.args.gpu), y.cuda(self.args.gpu)

                # Forward pass through the model
                out = self.model(x)

                feats = self.model.pre_logits(x)
                feats = feats.cpu().detach().tolist()
                for i in range(len(feats)):
                    features.append(feats[i])

                # Collect predictions and labels
                preds.append(torch.argmax(out, dim=1).cpu().detach().numpy())
                labels.append(y.cpu().detach().numpy())

            # Concatenate predictions and labels
            preds, labels = (np.concatenate(preds, axis=0), np.concatenate(labels, axis=0))
        return labels, preds, features

    @torch.no_grad()
    def evaluate(self, args=None):
        """
        Evaluate the model and calculate metrics.

        Args:
            eval_loader: DataLoader for evaluation. Default is None.
            args: Command-line arguments. Default is None.

        Returns:
            dict: Evaluation metrics.
        """
        print("Evaluating the model")
        self.model.eval()
        eval_loader = self.loader_dict['eval']

        labels, preds, _ = self.feedforward(eval_loader)
        metrics, _ = get_metrics(preds, labels, self.classes, tag="test/")
        print("Accuracy is : ", metrics["test/accuracy"])
        return metrics

    @torch.no_grad()
    def val(self, args=None):
        """
        Perform validation and calculate metrics based on the specified optimization criterion.

        Args:
            args (argparse.Namespace): Command-line arguments. Default is None.

        Returns:
            dict: Validation metrics.
        """
        # Set the model to evaluation mode
        self.model.eval()

        # Get the validation data loader
        val_loader = self.loader_dict['val']

        # Perform forward pass to get labels, predictions, and features
        model = self.model
        model.eval()
        labels, preds, features = self.feedforward(val_loader, return_feats=True)

        # Compute prototypes for each class
        prototypes = np.zeros((self.num_classes, self.num_feats))
        features = np.array(features)
        print("features shape", features.shape, features)
        for class_label in range(self.num_classes):
            # Extract mean features for the current class
            feats_for_class = features[labels == class_label]
            prototypes[class_label] = np.mean(feats_for_class, axis=0)

        # Get validation metrics and confusion matrix
        val_metrics, CM = get_metrics(preds, labels, self.classes, tag="val/")
        # print(val_metrics)

        # Choose optimization criterion based on command-line arguments
        if self.objective_name == "mean_recall":
            MR = MeanRecall(CM, prototypes, model, self.temperature)

        elif self.objective_name == "min_recall":
            MR = MinRecall(CM, prototypes, model, self.temperature,\
                           self.lambdas, self.beta, self.val_lr)
            self.lambdas = MR.lambdas

        elif self.objective_name == "min_HT_recall":
            MR = MinHTRecall(CM, prototypes, model, self.temperature,\
                           self.lambdas, self.beta, self.val_lr)
            self.lambdas = MR.lambdas

        elif self.objective_name == "mean_recall_min_coverage":
            MR = MeanRecallWithCoverage(CM, prototypes, model,\
                                        self.temperature, self.lambdas,\
                                        alpha=self.alpha, tau=self.tau,
                                        lambda_max=self.args.lambda_max)
            self.lambdas = MR.lambdas
        elif self.objective_name == "mean_recall_min_HT_coverage":
            MR = MeanRecallWithHTCoverage(CM, prototypes, model,\
                                          self.temperature, self.lambdas,\
                                          alpha=self.alpha, tau=self.tau,
                                          lambda_max=self.args.lambda_max)
            self.lambdas = MR.lambdas
        elif self.objective_name == "HM_min_HT_coverage":
            print("HM with HT coverage constraint")
            MR = HmeanWithHTCoverage(CM, prototypes, self.train_model,\
                                     self.temperature, self.lambdas,\
                                     alpha=self.alpha, tau=self.tau,
                                     lambda_max=self.args.lambda_max)
            self.lambdas = MR.lambdas
        elif self.objective_name == "g_mean":
            MR = Gmean(CM, prototypes, model, self.temperature)

        elif self.objective_name == "h_mean":
            MR = Hmean(CM, prototypes, model, self.temperature)

        elif self.objective_name == "HM_min_coverage":
            MR = HmeanWithCoverage(CM, prototypes, model,\
                                self.temperature,\
                                alpha=self.alpha, tau=self.tau,\
                                lambda_max=self.args.lambda_max)
            self.lambdas = MR.lambdas

        # Set the sampling distribution to sample from
        self.P = MR.P

        # Initialize FastJointSampler for Mixup
        JS = FastJointSampler(self.lb_dataset, self.ulb_dataset, model,
                              samp_dist=self.P, batch_size=args.batch_size, 
                              semi_supervised=False)

        # Set objective and update MixupSampler in loader_dict
        self.objective = MR
        self.loader_dict["MixupSampler"] = JS

        return val_metrics

    def save_model(self, save_name, save_path):
        """
        Save the SSL model.

        Args:
            save_name (str): Name of the saved model file.
            save_path (str): Directory path for saving the model.
        """
        save_filename = os.path.join(save_path, save_name)
        self.model = self.model.module if hasattr(self.model, 'module') else self.model
        torch.save({'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'it': self.it}, save_filename)

        print(f"Model saved: {save_filename}")

    def load_model(self, load_path):
        """
        Load a pre-trained SSL model.

        Args:
            load_path (str): Path to the pre-trained model checkpoint.
        """
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
