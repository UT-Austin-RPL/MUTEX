""" Vision Transformer (ViT) in PyTorch

A PyTorch implement of Vision Transformers as described in
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale' - https://arxiv.org/abs/2010.11929

The official jax code is released and available at https://github.com/google-research/vision_transformer

Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert

DeiT model defs and weights from https://github.com/facebookresearch/deit,
paper `DeiT: Data-efficient Image Transformers` - https://arxiv.org/abs/2012.12877

Hacked together by / Copyright 2020 Ross Wightman
"""
import os
import math
import logging
import random
import hashlib
import urllib
import warnings
import numpy as np
from tqdm import tqdm
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


from mutex.models.transformer.transformer_modules import *
from timm.models.layers import DropPath, trunc_normal_


_logger = logging.getLogger(__name__)

class SinusoidalPositionEncoding(nn.Module):
    def __init__(self,
                 input_size,
                 inv_freq_factor=10,
                 factor_ratio=None):
        super().__init__()
        self.input_size = input_size
        self.inv_freq_factor = inv_freq_factor
        channels = self.input_size
        channels = int(np.ceil(channels / 2) * 2)

        inv_freq = 1.0 / (
                self.inv_freq_factor ** (
                    torch.arange(0, channels, 2).float() / channels
                )
        )
        self.channels = channels
        self.register_buffer("inv_freq", inv_freq)

        if factor_ratio is None:
            self.factor = 1.
        else:
            factor = nn.Parameter(torch.ones(1) * factor_ratio)
            self.register_parameter("factor", factor)

    def forward(self, x):
        pos_x = torch.arange(
                x.shape[1], device=x.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)
        return emb_x * self.factor

class TemporalPoolTransformer(nn.Module):
    def __init__(self,
                 input_size,
                 num_layers,
                 num_heads,
                 head_output_size,
                 mlp_hidden_size,
                 dropout,
                 add_pool=True,
                 add_cross_modal_layer=False,
                 extra_hidden_layers=0,
                 **kwargs):
        super().__init__()

        self.layers = nn.ModuleList([])
        self.drop_path = DropPath(dropout) if dropout > 0. else nn.Identity()

        self.attention_output = {}
        self.add_pool = add_pool

        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleList([
                    Norm(input_size),
                    Attention(input_size,
                              num_heads=num_heads,
                              head_output_size=head_output_size,
                              dropout=dropout),
                    Norm(input_size),
                    TransformerFeedForwardNN(input_size,
                                             mlp_hidden_size,
                                             dropout=dropout)
                ])
            )

            self.attention_output[_] = None
        self.seq_len = None
        self.num_elements = None
        self.add_cross_modal_layer = add_cross_modal_layer
        self.extra_hidden_layers = extra_hidden_layers
        if self.add_cross_modal_layer:
            # add some finetuning layers with linear layer, relu, dropout
            #cross_modal_layer = [nn.Linear(input_size, input_size), nn.Dropout(p=0.1)]
            cross_modal_layers = [nn.Linear(input_size, input_size), nn.Dropout(p=0.1), nn.ReLU()]
            for _ in range(self.extra_hidden_layers):
                cross_modal_layers.append(nn.Linear(input_size, input_size))
                cross_modal_layers.append(nn.Dropout(p=0.1))
                cross_modal_layers.append(nn.ReLU())
            cross_modal_layers.append(nn.Linear(input_size, input_size))
            self.cross_modal_layer = nn.Sequential(*cross_modal_layers)
            # initialize with identity
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # initialize linear layers with identity
            m.weight.data.copy_(torch.eye(m.weight.size(0)))
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, mask=None):
        for layer_idx, (att_norm, att, ff_norm, ff) in enumerate(self.layers):
            if mask is not None:
                x = x + drop_path(att(att_norm(x), mask))
            else: # no masking, just use full attention
                x = x + drop_path(att(att_norm(x)))
            x = x + self.drop_path(ff(ff_norm(x)))
        if self.add_pool:
            if mask is not None:
                x = x * mask.unsqueeze(-1)
                # take mean over time
                x = torch.sum(x, dim=1, keepdim=True) / torch.sum(mask.unsqueeze(-1), dim=1, keepdim=True)
            else:
                x = torch.mean(x, dim=1, keepdim=True)
        if self.add_cross_modal_layer:
            x = self.cross_modal_layer(x)
            #if self.training:
            #    # print few weights of self.cross_modal_layer
            #    print("cross_modal_layer weights: ", self.cross_modal_layer[0].weight[0, :5])

        return x

    @property
    def device(self):
        return next(self.parameters()).device

class MLPTransform(nn.Module):
    def __init__(self,
                 input_size,
                 num_layers,
                 hidden_size,
                 output_size=None,
                 add_cross_modal_layer=False,
                 extra_hidden_layers=0,
        ):
        super().__init__()
        assert num_layers >= 1, "[error] num_layers < 1"
        if output_size is None:
            output_size = input_size
        sizes = [input_size] + [hidden_size] * (num_layers-1) + [output_size]
        layers = []
        for i in range(num_layers-1):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        self.projection = nn.Sequential(*layers)
        self.add_cross_modal_layer = add_cross_modal_layer
        self.extra_hidden_layers = extra_hidden_layers
        if self.add_cross_modal_layer:
            #cross_modal_layer = [nn.Linear(output_size, output_size), nn.Dropout(p=0.1)]
            cross_modal_layers = [nn.Linear(output_size, output_size), nn.Dropout(p=0.1), nn.ReLU()]
            for _ in range(self.extra_hidden_layers):
                cross_modal_layers.append(nn.Linear(output_size, output_size))
                cross_modal_layers.append(nn.Dropout(p=0.1))
                cross_modal_layers.append(nn.ReLU())
            cross_modal_layers.append(nn.Linear(output_size, output_size))
            self.cross_modal_layer = nn.Sequential(*cross_modal_layers)

    def forward(self, h, mask=None):
        """
        data:
            task_emb: (B, E)
        """
        h = self.projection(h) # (B, L, H)
        if mask is not None:
            h = h * mask.unsqueeze(-1)
            # take mean over time
            h = torch.sum(h, dim=1, keepdim=True) / torch.sum(mask.unsqueeze(-1), dim=1, keepdim=True)
        else:
            h = torch.mean(h, dim=1, keepdim=True)
        if self.add_cross_modal_layer:
            h = self.cross_modal_layer(h)
        return h

class TSEncoder(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """

    def __init__(
        self,
        input_size,
        drop_rate,
        max_frames,
        hidden_size,
        num_hidden_layers,
        output_size=None,
        vid_transform_kwargs=None,
        img_transform_kwargs=None,
        inst_transform_kwargs=None,
        gl_transform_kwargs=None,
        ai_transform_kwargs=None,
        ag_transform_kwargs=None):
        super(TSEncoder, self).__init__()

        self.input_size = input_size
        ## Defining the temporal embedding
        self.temporal_embed = SinusoidalPositionEncoding(input_size=input_size)
        self.ts_transform_modules = nn.ModuleDict({
            'vid_emb': eval(vid_transform_kwargs.network)(input_size=input_size, **vid_transform_kwargs.network_kwargs),
            'inst_emb': eval(inst_transform_kwargs.network)(input_size=input_size, **inst_transform_kwargs.network_kwargs),
            'ai_emb': eval(ai_transform_kwargs.network)(input_size=input_size, **ai_transform_kwargs.network_kwargs),
            'ag_emb': eval(ag_transform_kwargs.network)(input_size=input_size, **ag_transform_kwargs.network_kwargs),
            'img_emb': eval(img_transform_kwargs.network)(input_size=input_size, **img_transform_kwargs.network_kwargs),
            'gl_emb': eval(gl_transform_kwargs.network)(input_size=input_size, **gl_transform_kwargs.network_kwargs)})

        ## add mlp projection
        layers = [
            nn.LayerNorm(input_size),
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(drop_rate)]
        for _ in range(num_hidden_layers): layers.extend([nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.Dropout(drop_rate)])
        layers.extend([nn.Linear(hidden_size, output_size), nn.Dropout(drop_rate)])
        self.linear_projection = nn.Sequential(*layers)

        ## classification tokens
        self.lang_cls_token = nn.Parameter(torch.zeros(1, 1, input_size))
        self.inst_cls_token = nn.Parameter(torch.zeros(1, 1, input_size))
        self.gl_cls_token = nn.Parameter(torch.zeros(1, 1, input_size))
        self.image_cls_token = nn.Parameter(torch.zeros(1, 1, input_size))
        self.video_cls_token = nn.Parameter(torch.zeros(1, 1, input_size))
        self.ai_cls_token = nn.Parameter(torch.zeros(1, 1, input_size))
        self.ag_cls_token = nn.Parameter(torch.zeros(1, 1, input_size))

        ### Drops the features after concatenating with classification token
        #self.pos_drop = nn.Dropout(p=drop_rate)

        ## Initializing Parameter weights separately (?)
        trunc_normal_(self.lang_cls_token, std=0.002)
        trunc_normal_(self.inst_cls_token, std=0.002)
        trunc_normal_(self.gl_cls_token, std=0.002)
        trunc_normal_(self.image_cls_token, std=0.002)
        trunc_normal_(self.ai_cls_token, std=0.002)
        trunc_normal_(self.ag_cls_token, std=0.002)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def output_shape(self, input_shape):
        return input_shape

    def add_temporal_token(self, features):
        t_pos = self.temporal_embed(features).unsqueeze(dim=0) ## batch size
        features = features + t_pos
        return features

    def add_classification_tokens(
            self,
            lang_emb=None,
            inst_emb=None,
            gl_emb=None,
            img_emb=None,
            vid_emb=None,
            ag_emb=None,
            ai_emb=None,
            mode='add'):
        feat_dict = {}
        if not gl_emb is None:
            gl_cls_token = self.gl_cls_token.repeat((gl_emb.shape[0], 1, 1))
            gl_emb = gl_emb + gl_cls_token
        if not inst_emb is None:
            inst_cls_token = self.inst_cls_token.repeat((inst_emb.shape[0], 1, 1))
            inst_emb = inst_emb + inst_cls_token
        if not img_emb is None:
            image_cls_token = self.image_cls_token.repeat((img_emb.shape[0],1,1))
            img_emb = img_emb + image_cls_token
        if not vid_emb is None:
            video_cls_token = self.video_cls_token.repeat((vid_emb.shape[0],1,1))
            vid_emb = vid_emb + video_cls_token
        if ag_emb is not None:
            ag_cls_token = self.ag_cls_token.repeat((ag_emb.shape[0],1,1))
            ag_emb = ag_emb + ag_cls_token
        if ai_emb is not None:
            ai_cls_token = self.ai_cls_token.repeat((ai_emb.shape[0],1,1))
            ai_emb = ai_emb + ai_cls_token
        emb_dict = {'inst_emb': inst_emb, 'gl_emb': gl_emb, 'img_emb': img_emb, 'vid_emb': vid_emb, 'ag_emb': ag_emb, 'ai_emb': ai_emb}
        return emb_dict

    def aggregate_task_embs(
            self,
            emb_dict,
            emb_mask_dict,
            return_gt_rep=False,
            sg_gt_rep=False):
        emb_dict = self.add_classification_tokens(**emb_dict, mode='add')
        emb_list = []
        emb_mask_list = []
        gt_rep_key = 'vid_emb'
        gt_rep = None
        for k, v in emb_dict.items():
            if v is not None:
                x = self.ts_transform_modules[k](v, emb_mask_dict[k + '_mask'])
                # remove masks after transform modules.
                if sg_gt_rep and k == gt_rep_key:
                    x = x.detach()  # detach the gt_rep
                emb_list.append(x)
                if return_gt_rep and k == gt_rep_key:
                    gt_rep = x.clone().detach()

        emb = torch.cat(emb_list, dim=1)
        emb_mask = torch.cat(emb_mask_list, dim=1) if len(emb_mask_list) > 0 else None
        return emb, emb_mask, gt_rep

    def forward(self, task_emb):
        task_emb = self.linear_projection(task_emb)
        return task_emb
