import kornia
import numpy as np
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from copy import deepcopy
from torch.utils.data import ConcatDataset, DataLoader

from mutex.algos.base import Sequential
from mutex.models.policy import *
from mutex.utils import *
from mutex.metric import *


class Multitask(Sequential):
    """
    The multitask learning baseline/upperbound.
    """
    def __init__(self,
                 n_tasks,
                 cfg,
                 logger,
                 **policy_kwargs):
        super().__init__(n_tasks=n_tasks, cfg=cfg, logger=logger, **policy_kwargs)
        print('# Actor Learnable Params: %d' % sum(p.numel() for p in self.policy.parameters() if p.requires_grad))

    def train_loop(self, dataloader, epoch):
        self.policy.train()
        training_loss = 0.
        train_log_dict = {'epoch': epoch}

        for (idx, data) in tqdm(enumerate(dataloader), total=len(dataloader), disable=self.cfg.train.debug):
            loss, log_info = self.observe(data)
            training_loss += loss
            train_log_dict = update_log_dict(train_log_dict, log_info, prefix='train', mode='run_avg')
            if self.cfg.train.debug:
                print("[debug] Train loss:", loss)
        training_loss /= len(dataloader)
        self.policy.anneal_weights(epoch)

        return training_loss, train_log_dict

    def val_loop(self, dataloader, epoch):
        self.policy.eval()
        val_log_dict = {'epoch': epoch}
        val_loss = 0.
        for (idx, data) in tqdm(enumerate(dataloader), total=len(dataloader), disable=self.cfg.train.debug):
            loss, log_info = self.eval_observe(data)
            val_loss += loss
            val_log_dict = update_log_dict(val_log_dict, log_info, prefix='val', mode='run_avg')
            if self.cfg.train.debug:
                print("[debug] Validation loss:", loss)
        val_loss /= len(dataloader)

        return val_loss, val_log_dict

    def add_misc_log(self, log_dict):
        lr_list = self.scheduler.get_last_lr()
        lr_dict = {}
        for ind, lr in enumerate(lr_list):
            lr_dict[f'train/lr_{ind:03d}'] = lr

        log_dict.update(lr_dict)
        return log_dict

    def learn_all_tasks(self, datasets, start_epoch=0, model_prefix=''):
        concat_dataset = ConcatDataset(datasets)
        train_dataset, val_dataset = torch.utils.data.random_split(concat_dataset, [int(self.cfg.data.train_dataset_ratio*len(concat_dataset)),len(concat_dataset)-int(self.cfg.data.train_dataset_ratio*len(concat_dataset))])

        # learn on all tasks, only used in multitask learning
        model_checkpoint_name = os.path.join(self.model_dir,
                                             f"{model_prefix}multitask_model.pth")

        train_dataloader = DataLoader(train_dataset,
                                      batch_size=self.cfg.train.batch_size,
                                      num_workers=self.cfg.train.num_workers,
                                      shuffle=True)
        val_dataloader = DataLoader(val_dataset,
                                    batch_size=self.cfg.train.batch_size,
                                    num_workers=self.cfg.train.num_workers,
                                    shuffle=True)

        best_val_loss = float(1e6)
        prev_success_rate = -1.0
        best_state_dict = self.policy.state_dict() # currently save the best model

        # for evaluate how fast the agent learns on current task, this corresponds
        # to the area under success rate curve on the new task.
        cumulated_counter = 0.0
        idx_at_best_succ = 0
        training_loss = None
        successes = []
        losses = []
        val_losses = []

        # start training
        # Note: last epoch is only for validation run
        for epoch in range(start_epoch, self.cfg.train.n_epochs+1, 1):
            log_dict = {}
            print(f"[info] Epoch: {epoch:3d}")

            if (not self.cfg.train.debug) and \
                    ((epoch % self.cfg.train.val_every == 0) or (epoch == self.cfg.train.n_epochs)):  ## Validation loop
                t0 = time.time()
                validation_loss, val_log_dict = self.val_loop(val_dataloader, epoch=epoch)
                log_dict.update(val_log_dict)
                val_losses.append(validation_loss)
                t1 = time.time()
                if self.save_best_model and (best_val_loss > validation_loss):
                    torch_save_model(self.policy, model_checkpoint_name, cfg=self.cfg)
                    best_val_loss = validation_loss

                if self.save_ckpt:
                    self.save_checkpoint(epoch=epoch)
                print(f"[info] Epoch: {epoch:3d} | val loss: {validation_loss:5.2f} | best val loss: {best_val_loss:5.2f} | time: {(t1-t0)/60:4.2f}")

                model_checkpoint_name_ep = os.path.join(self.model_dir,
                                                        f"{model_prefix}multitask_model_ep{epoch:03d}.pth")
                print(f"[info] Saving model to {model_checkpoint_name_ep}")
                torch_save_model(self.policy, model_checkpoint_name_ep, cfg=self.cfg)
                losses.append(training_loss)
                torch.save(np.array(losses), os.path.join(self.experiment_dir, f"multitask_auc.log"))
                torch.save(np.array(val_losses), os.path.join(self.experiment_dir, f"multitask_val_auc.log"))
                if epoch == self.cfg.train.n_epochs:
                    break # stop training after last validation run

            t0 = time.time()
            training_loss, train_log_dict = self.train_loop(train_dataloader, epoch=epoch)
            t1 = time.time()
            log_dict.update(train_log_dict)

            print(f"[info] Epoch: {epoch:3d} | train loss: {training_loss:5.2f} | time: {(t1-t0)/60:4.2f}")

            log_dict = self.add_misc_log(log_dict)
            self.update_logger(log_dict)

            if self.scheduler is not None and epoch > 0:
                self.scheduler.step()

        return 0.0, 0.0
