import gc
import copy
import gc
import cv2
import imageio
import numpy as np
import os
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
import time
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from robosuite import load_controller_config
from time import gmtime, strftime
from torch.multiprocessing import Array
from torch.utils.data import DataLoader
from tqdm import trange

from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv
from mutex.utils import *


def raw_obs_to_tensor_obs(obs, task_emb, cfg):
    """
        Prepare the tensor observations as input for the algorithm.
    """
    env_num = len(obs)
    if len(task_emb.shape) == 1: ## adds a new dimension and repeats along it
        task_emb = task_emb.repeat(env_num, 1)
    elif len(task_emb.shape) == 2:
        task_emb = task_emb.repeat(env_num, 1, 1)
    else:
        raise NotImplementedError

    data = {
        "obs": {
            "agentview_rgb"  : [],
            "eye_in_hand_rgb": [],
            "gripper_states" : [],
            "joint_states"   : [],
        },
        "task_emb": task_emb,
    }

    for k in range(env_num):
        data["obs"]["agentview_rgb"].append(ObsUtils.process_obs(
                torch.from_numpy(obs[k]["agentview_image"]),
                obs_key="agentview_rgb"))

        data["obs"]["eye_in_hand_rgb"].append(ObsUtils.process_obs(
                torch.from_numpy(obs[k]["robot0_eye_in_hand_image"]),
                obs_key="eye_in_hand_rgb"))

        data["obs"]["gripper_states"].append(torch.from_numpy(np.array(
                obs[k]["robot0_gripper_qpos"])).float()),

        data["obs"]["joint_states"].append(torch.from_numpy(np.array(
                obs[k]["robot0_joint_pos"])).float()),

    for key in data["obs"]:
        data["obs"][key] = torch.stack(data["obs"][key])

    data = TensorUtils.map_tensor(data,
                                  lambda x: safe_device(x, device=cfg.device))
    return data


def evaluate_one_task_success(
                        cfg,
                        algo,
                        task,
                        task_emb,
                        task_id,
                        sim_states=None,
                        task_str="",
    ):
    """
        Evaluate a single task's success rate
        sim_states: if not None, will keep track of all simulated states during
                    evaluation, mainly for visualization and debugging purpose
        task_str:   the key to access sim_states dictionary
    """

    t0 = time.time()
    algo.eval()

    # initiate evaluation envs
    env_args = {
        "bddl_file_name": os.path.join(cfg.bddl_folder, task.problem_folder, task.bddl_file),
        "camera_heights": cfg.data.img_h,
        "camera_widths": cfg.data.img_w,
    }

    env_num = min(cfg.eval.num_procs, cfg.eval.n_eval) if cfg.eval.use_mp else 1
    eval_loop_num = (cfg.eval.n_eval + env_num - 1) // env_num

    if env_num == 1:
        env = OffScreenRenderEnv(**env_args)
    else:
        env = SubprocVectorEnv([
            lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)])
    env.seed(cfg.seed)

    # Evaluation loop
    num_success = 0

    # get fixed init states to control the experiment randomness
    init_states_path = os.path.join(cfg.init_states_folder,
                                    task.problem_folder,
                                    task.init_states_file)
    init_states = torch.load(init_states_path)

    for i in range(eval_loop_num):
        env.reset()

        indices = np.arange(i*env_num, (i+1)*env_num) % init_states.shape[0]
        init_states_ = init_states[indices]

        dones = [False] * env_num
        steps = 0
        algo.reset()
        obs = env.set_init_state(init_states_) if env_num > 1 else env.set_init_state(init_states_[0])

        # dummy actions [env_num, 7] all zeros for initial physics simulation
        dummy = np.zeros((env_num, 7)) if env_num > 1 else np.zeros((7,))
        for _ in range(5):
            obs, _, _, _ = env.step(dummy)

        for k in range(env_num):
            task_emb_eval = task_emb[(i*env_num+k) % task_emb.shape[0]]

        if task_str != "":
            sim_state = env.get_sim_state()
            for k in range(env_num):
                if i*env_num+k < cfg.eval.n_eval:
                    sim_states[i*env_num+k].append(sim_state[k])

        while steps < cfg.eval.max_steps:
            steps += 1

            if env_num == 1: obs = [obs]
            data = raw_obs_to_tensor_obs(obs, task_emb_eval, cfg)

            actions = algo.policy.get_action(data)
            if env_num == 1: actions = actions[0]

            obs, reward, done, info = env.step(actions)

            # record the sim states for replay purpose
            if task_str != "":
                sim_state = env.get_sim_state()
                for k in range(env_num):
                    if i*env_num+k < cfg.eval.n_eval:
                        sim_states[i*env_num+k].append(sim_state[k])

            # check whether succeed
            if env_num == 1:
                dones[0] = done
            else:
                for k in range(env_num):
                    dones[k] = dones[k] or done[k]

            if all(dones): break

        # a new form of success record
        for k in range(env_num):
            if i*env_num+k < cfg.eval.n_eval:
                num_success += int(dones[k])

    success_rate = num_success / cfg.eval.n_eval
    env.close()
    gc.collect()
    t1 = time.time()
    print(f"[info] evaluate task {task_id} takes {(t1-t0)/60:.1f} min")
    return success_rate, sim_states


def evaluate_success(cfg, algo, benchmark, task_ids, result_summary=None):
    """
        Evaluate the success rate for all task in task_ids.
    """
    algo.eval()
    successes = []
    for i in task_ids:
        task_i = benchmark.get_task(i)
        task_emb = benchmark.get_task_emb(i)
        task_str = f"k{task_ids[-1]}_p{i}"
        curr_summary = result_summary[task_str] if result_summary is not None else None
        success_rate = evaluate_one_task_success(cfg,
                                                 algo,
                                                 task_i,
                                                 task_emb,
                                                 i,
                                                 sim_states=curr_summary,
                                                 task_str=task_str)
        successes.append(success_rate)
    return np.array(successes)


def evaluate_multitask_training_success(cfg, algo, benchmark, task_ids, result_summary=None):
    """
        Evaluate the success rate for all task in task_ids.
    """
    algo.eval()
    successes = []
    for i in task_ids:
        task_i = benchmark.get_task(i)
        task_emb = benchmark.get_task_emb(i) # [num_eval_ts, T, E]
        task_str = f"k{task_ids[-1]}_p{i}"
        curr_summary = {}
        for eval_traj_i in range(cfg.eval.n_eval):
            curr_summary[eval_traj_i] = []

        success_rate, curr_summary = evaluate_one_task_success(cfg=cfg,
                                                 algo=algo,
                                                 task=task_i,
                                                 task_emb=task_emb,
                                                 task_id=i,
                                                 task_str=task_str,
                                                 sim_states=curr_summary)
        successes.append(success_rate)
        print(f"Task {task_i.name}; Success Rate: {success_rate}")
        result_summary[i].update({'sim_states': copy.deepcopy(curr_summary)})
    return np.array(successes), result_summary


@torch.no_grad()
def evaluate_loss(cfg, algo, benchmark, datasets):
    """
        Evaluate the loss on all datasets.
    """
    algo.eval()
    losses = []
    for i, dataset in enumerate(datasets):
        dataloader = DataLoader(dataset,
                                batch_size=cfg.eval.batch_size,
                                num_workers=cfg.eval.num_workers,
                                shuffle=False)
        test_loss = 0
        for data in dataloader:
            data = TensorUtils.map_tensor(
                    data, lambda x: safe_device(x, device=cfg.device))
            loss = algo.policy.get_loss(data)
            test_loss += loss.item()
        test_loss /= len(dataloader)
        losses.append(test_loss)
    return np.array(losses)
