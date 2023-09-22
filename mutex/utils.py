import os
import copy
import cv2
import json
import numpy as np
import random
import robomimic.utils.tensor_utils as TensorUtils
import torch
import torch.nn as nn
import torchvision

from easydict import EasyDict
from thop import profile
from torch.utils.data import ConcatDataset, DataLoader

def set_params(cfg):
    print(cfg.policy.policy_type)
    if cfg.model_type == 'S':
        cfg.policy.decoder.network_kwargs.depth = 2
        cfg.policy.decoder.network_kwargs.perceiver_ct_index = [0]
        cfg.policy.transformer_cross_attn_ind = [0, 2, 4]
        cfg.policy.transformer_num_layers = 5
    elif cfg.model_type == 'M':
        cfg.policy.decoder.network_kwargs.depth = 3
        cfg.policy.decoder.network_kwargs.perceiver_ct_index = [0, 1, 2]
        cfg.policy.transformer_cross_attn_ind = [0, 2, 4]
        cfg.policy.transformer_num_layers = 5
    elif (cfg.model_type == 'L1') or (cfg.model_type == 'L'):
        cfg.policy.decoder.network_kwargs.depth = 4
        cfg.policy.decoder.network_kwargs.perceiver_ct_index = [0, 1, 2, 3, 4]
        cfg.policy.transformer_cross_attn_ind = [0, 2, 4, 6]
        cfg.policy.transformer_num_layers = 7
    else:
        raise NotImplementedError
    return cfg

def save_video(imgs, video_path, duration=-1, fps=60):
    """
    Save a video from a list of images
    :param imgs: list of images
    :param video_path: path to save the video
    :param duration: duration of the video in seconds
    :param fps: frames per second
    :return:
    """
    if duration > 0:
        if len(imgs) < duration * fps:
            for _ in range(duration * fps - len(imgs)):
                imgs.append(imgs[-1])
        elif len(imgs) > duration * fps:
            imgs = imgs[:duration * fps]
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(video_path, fourcc, fps, (imgs[0].shape[1], imgs[0].shape[0]))
    for i in range(len(imgs)):
        out.write(imgs[i])
    out.release()

def save_img(img, img_path):
    """
    Save an image
    :param img: image
    :param img_path: path to save the image
    :return:
    """
    cv2.imwrite(img_path, img)


def _input(message, input_type=str, valid_values=['0','1']): ## Custom input message format
    while True:
        try:
            inp = input_type(input(message))
            assert inp in valid_values
            return inp
        except:
            pass

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def update_log_dict(logger_dict, info, prefix, mode='run_avg'):
    for k,v in info.items():
        log_key = prefix + '/' + k
        if not log_key in logger_dict:
            logger_dict[log_key] = v
            logger_dict[log_key+'_num'] = 1
        else:
            if mode == 'run_avg':
                prev_num = logger_dict[log_key+'_num']
                logger_dict[log_key] = (logger_dict[log_key]*prev_num + v)/(prev_num + 1)
            else:
                logger_dict[log_key] = logger_dict[log_key]
            logger_dict[log_key+'_num'] += 1
    return logger_dict

def set_requires_grad(model, mode):
    for p in model.parameters():
        p.requires_grad = mode

def sample_frames(num_frames, vlen, sample='rand', fix_start=None):
    acc_samples = min(num_frames, vlen)
    intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
    ranges = []
    for idx, interv in enumerate(intervals[:-1]):
        ranges.append((interv, intervals[idx + 1] - 1))
    if sample == 'rand':
        frame_idxs = [random.choice(range(x[0], x[1])) for x in ranges]
    elif fix_start is not None:
        frame_idxs = [x[0] + fix_start for x in ranges]
    elif sample == 'uniform':
        frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]
    else:
        raise NotImplementedError

    return frame_idxs

def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def control_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def safe_device(x, device="cpu"):
    if device == "cpu":
        return x.cpu()
    elif "cuda" in device:
        if torch.cuda.is_available():
            return x.to(device)
        else:
            return x.cpu()


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def torch_save_model(model, model_path, cfg=None, previous_masks=None):
    torch.save({"state_dict": model.state_dict(),
                "cfg": cfg,
                "previous_masks": previous_masks}, model_path)


def torch_load_model(model_path, device='cuda'):
    # Load the state dict from a saved checkpoint
    model_dict = torch.load(model_path, map_location=device)
    # Create a new state dict without the unwanted weights
    new_state_dict = {}
    for key, value in model_dict["state_dict"].items():
        name = key[7:] if key.startswith('module.') else key
        if not name.startswith('visual_emb_model.pre_compute_vision_model_encoder'):
            new_state_dict[name] = value

    cfg = None
    if "cfg" in model_dict:
        cfg = model_dict["cfg"]
    if "previous_masks" in model_dict:
        previous_masks = model_dict["previous_masks"]
    return new_state_dict, cfg, previous_masks


def get_train_test_loader(dataset, train_ratio, train_batch_size, test_batch_size, num_workers=(0, 0)):
    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=train_batch_size,
                                  num_workers=num_workers[0],
                                  shuffle=True)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=test_batch_size,
                                 num_workers=num_workers[1],
                                 shuffle=False)
    return train_dataloader, test_dataloader


def confidence_interval(p, n):
    return 1.96 * np.sqrt(p*(1-p) / n)


def compute_IoU(A, B, threshold):
    # A: an array with entries in [0, 1]
    # B: tensor with the same shape as A (groud-truth segmentation):
    #   1 for obj_of_interest
    #  -1 for robot
    #   0 for other things
    # threshold: (A>threshold) will be set to 1.
    bch = A.shape[0]
    assert A.shape == B.shape, "[error] shape of A and B should be same for IoU"
    mA = np.logical_and((A > threshold), (B >= 0))

    intersection = np.logical_and(mA, (B > 0)).reshape(bch, -1).sum(-1)
    union = np.logical_or(mA, (B > 0)).reshape(bch, -1).sum(-1)

    IoU = intersection / (union + 1e-8)
    return IoU # (bch,)


def compute_flops(algo, dataset, cfg):
    model = copy.deepcopy(algo.policy)
    tmp_loader = DataLoader(dataset,
                            batch_size=1,
                            num_workers=0,
                            shuffle=True)
    data = next(iter(tmp_loader))
    data = TensorUtils.map_tensor(data,
                                  lambda x: safe_device(x, device=cfg.device))
    macs, params = profile(model, inputs=(data,), verbose=False)
    GFLOPs = macs * 2 / 1e9
    MParams = params / 1e6
    del model
    return GFLOPs, MParams
