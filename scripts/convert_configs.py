import os
import json
# OmegaConfg and EasyDict
from omegaconf import OmegaConf
from easydict import EasyDict as edict
import torch

from mutex.utils import NpEncoder

def cleanup_configs(cfg):
    # add rep_loss_coef
    if 'rep_loss_coef' not in cfg.policy:
        cfg.policy.rep_loss_coef = 0.1
    # add add_rep_loss
    if 'add_rep_loss' not in cfg.policy:
        cfg.policy.add_rep_loss = False
    # add sg_gt_rep
    if 'sg_gt_rep' not in cfg.policy:
        cfg.policy.sg_gt_rep = False

    # copy the language_encoder to projection_layer
    cfg.policy.projection_layer = cfg.policy.language_encoder

    # remove language_encoder
    cfg.policy.pop('language_encoder', None)
    # remove cfg.policy.multimodal
    if 'multimodal' in cfg.policy:
        cfg.policy.pop('multimodal', None)
    # remove cfg.policy.add_regularizer_loss
    if 'add_regularizer_loss' in cfg.policy:
        cfg.policy.pop('add_regularizer_loss', None)
    # remove add_extra_attn variable
    if 'add_extra_attn' in cfg.policy.projection_layer:
        cfg.policy.projection_layer.pop('add_extra_attn', None)
    # remove this policy.projection_layer.network_kwargs.extra_attn_kwargs
    if 'extra_attn_kwargs' in cfg.policy.projection_layer.network_kwargs:
        cfg.policy.projection_layer.network_kwargs.pop('extra_attn_kwargs', None)
    # remove cfg.policy.seq_rep_loss
    if 'seq_rep_loss' in cfg.policy:
        cfg.policy.pop('seq_rep_loss', None)
    # remove policy.ts_transform \
    if 'ts_transform' in cfg.policy:
        cfg.policy.pop('ts_transform', None)

    # change BCPerceiverIOPolicy to BCMutexPolicy
    cfg.policy.policy_type = 'BCMutexPolicy'
    # change PerceiverDecoder to TransformerCrossDecoder
    cfg.policy.decoder.network = 'TransformerCrossDecoder'
    # change TokenAdder to TSEncoder
    cfg.policy.projection_layer.network = 'TSEncoder'
    return cfg

def main():
    config_path = 'release_exps/mutex/config.json'
    config_old_path = 'release_exps/mutex/config_old.json'
    with open(config_path, "r") as f:
        cfg = json.load(f)

    ## preprocessing
    cfg = edict(cfg)
    cfg = cleanup_configs(cfg)
    # copy config_path tp config_old.json
    os.system('cp {} {}'.format(config_path, config_old_path))
    # save the new config
    # note that config is in edict format
    with open(config_path, "w") as f:
        json.dump(cfg, f, cls=NpEncoder, indent=4)


if __name__ == '__main__':
    main()
