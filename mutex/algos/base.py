import cv2
import numpy as np
import os
import robomimic.utils.tensor_utils as TensorUtils
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from hydra.utils import get_original_cwd, to_absolute_path
from torch.utils.data import ConcatDataset, DataLoader

from mutex.models.policy import *
from mutex.utils import *
from mutex.metric import *

class DPWrapper(nn.DataParallel): ## Simple DDP wrapper to access attributes of policy class
    @property
    def log_info(self):
        return self.module.log_info

    def stop_masked_losses(self, stop_bool):
        return self.module.stop_masked_losses(stop_bool)

    def stop_matching_loss(self, stop_bool):
        return self.module.stop_matching_loss(stop_bool)

    def anneal_weights(self, epoch):
        return self.module.anneal_weights(epoch)

class Sequential(nn.Module):
    """
    The sequential BC baseline.
    """
    def __init__(self, n_tasks, cfg, logger=None):
        super().__init__()
        self.cfg = cfg
        self.logger = logger
        self.epoch = 0
        self.loss_scale = cfg.train.loss_scale
        self.n_tasks = n_tasks
        self.experiment_dir = cfg.experiment_dir
        self.model_dir = os.path.join(cfg.experiment_dir, "models")
        self.ckpt_dir = os.path.join(cfg.experiment_dir, "ckpt")
        self.algo = cfg.lifelong.algo
        self.save_ckpt = True
        self.save_best_model = True
        if 'save_ckpt' in cfg:
            self.save_ckpt = cfg.save_ckpt
        if 'save_best_model' in cfg:
            self.save_best_model = cfg.save_best_model

        self.policy = eval(cfg.policy.policy_type)(cfg, cfg.shape_meta)
        if self.cfg.num_gpus > 1:
            print("Using DataParallel")
            self.policy = DPWrapper(self.policy)
        self.current_task = -1

        make_dir(self.model_dir)
        make_dir(self.ckpt_dir)

    def end_task(self, dataset, task_id, benchmark, env=None):
        pass

    def start_task(self, task):
        self.current_task = task

        # initialize the optimizer and scheduler
        self.optimizer = eval(self.cfg.train.optimizer.name)(
                self.policy.parameters(),
                **self.cfg.train.optimizer.kwargs)

        self.scheduler = None
        T_max = 40 if self.cfg.train.n_epochs == 20 else self.cfg.train.n_epochs
        if self.cfg.train.scheduler is not None:
            self.scheduler = eval(self.cfg.train.scheduler.name)(
                    self.optimizer,
                    T_max=T_max,
                    **self.cfg.train.scheduler.kwargs)

    def observe(self, data):
        assert self.policy.training == True
        data = TensorUtils.map_tensor(
                data, lambda x: safe_device(x, device=self.cfg.device))
        self.optimizer.zero_grad()
        loss = self.policy(data)
        log_info = self.policy.log_info
        if isinstance(self.policy, nn.DataParallel):
            loss = loss.mean()

        (self.loss_scale*loss).backward()
        if self.cfg.train.grad_clip is not None:
            grad_norm = nn.utils.clip_grad_norm_(self.policy.parameters(),
                                                 self.cfg.train.grad_clip)
        self.optimizer.step()
        return loss.item(), log_info

    def eval_observe(self, data):
        assert self.policy.training == False
        data = TensorUtils.map_tensor(
                data, lambda x: safe_device(x, device=self.cfg.device))
        with torch.no_grad():
            loss = self.policy(data)
            if isinstance(self.policy, nn.DataParallel):
                loss = loss.mean()
            log_info = self.policy.log_info
        return loss.item(), log_info

    def learn_one_task(self, dataset, task_id, benchmark, result_summary):

        self.start_task(task_id)

        # recover the corresponding manipulation task ids
        gsz = self.cfg.data.task_group_size
        manip_task_ids = list(range(task_id*gsz, (task_id+1)*gsz))

        model_checkpoint_name = os.path.join(self.model_dir,
                                             f"task{task_id}_model.pth")

        train_dataloader = DataLoader(dataset,
                                      batch_size=self.cfg.train.batch_size,
                                      num_workers=self.cfg.train.num_workers,
                                      shuffle=True)

        prev_success_rate = -1.0
        best_state_dict = self.policy.state_dict() # currently save the best model

        # for evaluate how fast the agent learns on current task, this corresponds
        # to the area under success rate curve on the new task.
        cumulated_counter = 0.0
        idx_at_best_succ = 0
        successes = []
        losses = []

        #### ========== creat the environment once ==================

        task = benchmark.get_task(task_id)
        task_emb = benchmark.get_task_emb(task_id)

        ## initiate evaluation envs
        #cfg = self.cfg
        #env_args = {
        #    "bddl_file_name": os.path.join(cfg.bddl_folder, task.problem_folder, task.bddl_file),
        #    "camera_heights": cfg.data.img_h,
        #    "camera_widths": cfg.data.img_w,
        #}

        #env_num = min(cfg.eval.num_procs, cfg.eval.n_eval) if cfg.eval.use_mp else 1
        #eval_loop_num = (cfg.eval.n_eval + env_num - 1) // env_num

        #env = SubprocVectorEnv([
        #    lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)])
        #env.seed(cfg.seed)

        # start training
        for epoch in range(0, self.cfg.train.n_epochs+1):

            t0 = time.time()
            if epoch > 0: # update
                self.policy.train()
                training_loss = 0.
                for (idx, data) in enumerate(train_dataloader):
                    loss = self.observe(data)
                    training_loss += loss
                training_loss /= len(train_dataloader)
            else: # just evaluate the zero-shot performance on 0-th epoch
                training_loss = 0.
                for (idx, data) in enumerate(train_dataloader):
                    loss = self.eval_observe(data)
                    training_loss += loss
                training_loss /= len(train_dataloader)
            t1 = time.time()

            print(f"[info] Epoch: {epoch:3d} | train loss: {training_loss:5.2f} | time: {(t1-t0)/60:4.2f}")

            # TODO(@Bo) Find a better solution, it is caused by the num_workers in dataloader
            time.sleep(0.1)

            if epoch % self.cfg.eval.eval_every == 0: # evaluate BC loss
                self.policy.eval()
                losses.append(training_loss)

                t0 = time.time()

                #success_rates = evaluate_multitask_training_success(self.cfg,
                #                                                    self,
                #                                                    benchmark,
                #                                                    task_ids=manip_task_ids)
                #success_rate = success_rates.mean()
                task_str = f"k{task_id}_e{epoch//self.cfg.eval.eval_every}"
                success_rate = evaluate_one_task_success(self.cfg,
                                                         self,
                                                         task,
                                                         task_emb,
                                                         task_id,
                                                         sim_states=result_summary[task_str],
                                                         task_str=task_str if self.cfg.eval.debug else "")
                successes.append(success_rate)

                if prev_success_rate < success_rate:
                    torch_save_model(self.policy, model_checkpoint_name, cfg=self.cfg)
                    prev_success_rate = success_rate
                    idx_at_best_succ = len(losses) - 1

                t1 = time.time()

                cumulated_counter += 1.0

                ci = confidence_interval(success_rate, self.cfg.eval.n_eval)

                tmp_successes = np.array(successes)
                tmp_successes[idx_at_best_succ:] = successes[idx_at_best_succ]
                print(f"[info] Epoch: {epoch:3d} | succ: {success_rate:4.2f} ± {ci:4.2f} | best succ: {prev_success_rate} " + \
                        f"| succ. AoC {tmp_successes.sum()/cumulated_counter:4.2f} | time: {(t1-t0)/60:4.2f}", flush=True)

            if self.scheduler is not None and epoch > 0:
                self.scheduler.step()

        self.policy.load_state_dict(torch_load_model(model_checkpoint_name)[0])

        if self.cfg.lifelong.algo == "PackNet": # need preprocess weights for PackNet
            self.end_task(dataset, task_id, benchmark)
        else:
            self.end_task(dataset, task_id, benchmark)

        # return the metrics regarding forward transfer
        loss_at_best_succ = losses[idx_at_best_succ]
        success_at_best_succ = successes[idx_at_best_succ]

        losses = np.array(losses)
        successes = np.array(successes)
        auc_checkpoint_name = os.path.join(self.experiment_dir,
                                             f"task{task_id}_auc.log")
        torch.save({
            "success": successes,
            "loss": losses,}, auc_checkpoint_name)

        losses[idx_at_best_succ:] = loss_at_best_succ
        successes[idx_at_best_succ:] = success_at_best_succ
        return successes.sum() / cumulated_counter, losses.sum() / cumulated_counter

    def reset(self):
        self.policy.reset()

    def save_checkpoint(self, epoch):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            #'loss': LOSS,
        }, os.path.join(self.ckpt_dir, 'last_ckpt.pt'))

    def load_checkpoint(self):
        ckpt=torch.load(os.path.join(self.ckpt_dir, 'last_ckpt.pt'))
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        self.load_state_dict(ckpt['model_state_dict'])
        return ckpt['epoch']

    def update_logger(self, log_dict):
        if not self.logger is None:
            # remove keys that end with '_num'
            log_dict = {k: v for k, v in log_dict.items() if not k.endswith('_num')}
            self.logger.log(log_dict)
