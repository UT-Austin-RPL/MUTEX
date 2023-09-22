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
"""
Mutex Adopted this code from:
Tang, Zineng, et al. "Perceiver-vl: Efficient vision-and-language modeling with iterative latent attention." Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. 2023.
"""
import os
import copy
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
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x)
        bs, num_q = x.shape
        pos_x = x.reshape(-1).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)
        emb_x = emb_x.reshape(bs, num_q, -1)
        return emb_x * self.factor

    def output_shape(self, input_shape):
        return input_shape

    def output_size(self, input_size):
        return input_size

class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        q_dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        assert q_dim % num_heads == 0, "The latent_embed_dim should be divisible by num_heads. Currently {}, {}".format(q_dim, num_heads)
        assert dim % num_heads == 0, "The embed_dim (from context) should be divisible by num_heads. Currently {}, {}".format(dim, num_heads)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(q_dim, q_dim, bias=True)
        self.k = nn.Linear(dim, q_dim, bias=True)
        self.v = nn.Linear(dim, q_dim, bias=True)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(q_dim, q_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context=None, mask=None):
        skip_x = x
        B, N, C = x.shape
        q = self.q(x).reshape(B, x.size(1), self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(context).reshape(B, context.size(1), self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(context).reshape(B, context.size(1), self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            mask = mask.bool()
            if len(mask.shape) == 2: # (B, N)
                attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))
            elif len(mask.shape) == 3 and mask.shape[0] == 1: # (1, N, N)
                attn = attn.masked_fill(~mask[None, :, :, :], float("-inf"))
            else:
                raise Exception("mask shape is not correct for attention")
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, skip_x, attn


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape

        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            mask = mask.bool()
            if len(mask.shape) == 2: # (B, N)
                attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))
            elif len(mask.shape) == 3 and mask.shape[0] == 1: # (1, N, N)
                attn = attn.masked_fill(~mask[None, :, :, :], float("-inf"))
            else:
                raise Exception("mask shape is not correct for attention")
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        use_context=False,
        post_norm=False,
        q_dim=None,
    ):
        super().__init__()
        if q_dim is None: ## Separate encoding dimension size for q for cross attention
            q_dim = dim
        self.norm1 = norm_layer(q_dim)
        if use_context:
            assert not q_dim is None
            self.attn = CrossAttention(
                dim,
                q_dim=q_dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
        else:
            assert q_dim == dim
            self.attn = Attention(
                q_dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(q_dim)
        mlp_hidden_dim = int(q_dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=q_dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        self.post_norm = post_norm
        if post_norm:
            self.post_norm_layer = norm_layer(q_dim)

    def forward(self, x, context=None, mask=None):

        if context is None:
            _x, attn = self.attn(self.norm1(x), mask=mask)
        else:
            _x, x, attn = self.attn(self.norm1(x), context=context, mask=mask)
        x = x + self.drop_path(_x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if self.post_norm:
            self.post_norm_layer(x)
        return x, attn


class TransformerCrossDecoder(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """

    def __init__(
        self,
        input_size=768,
        output_size=128,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        perceiver_ct_index=[0, 4, 8],
        norm_layer=None,
        num_action_q=10,
        max_text_q=25,  # Unused
        max_region_q=50,  # Unused
        max_frame_q=32,  # Unused
        max_instruct_q=10,
        query_drop=False,
        query_norm=None,
    ):
        """
        Args:
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            hybrid_backbone (nn.Module): CNN backbone to use in-place of PatchEmbed module
            norm_layer: (nn.Module): normalization layer
        """
        super(TransformerCrossDecoder, self).__init__()

        embed_dim = input_size
        latent_embed_dim = output_size
        self.embed_dim = embed_dim  # num_features for consistency with other models
        self.latent_embed_dim = latent_embed_dim

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        ## Attention Blocks
        self.depth = depth
        self.cross_layers_visual = perceiver_ct_index
        num_cross_blocks = len(self.cross_layers_visual)
        self.crossatt_blocks_visual = nn.ModuleList([
                Block(
                    dim=embed_dim,
                    q_dim=latent_embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    norm_layer=norm_layer,
                    use_context=True,
                ) for i in range(num_cross_blocks)])
        self.blocks = nn.ModuleList([
                Block(
                    dim=latent_embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    norm_layer=norm_layer,
                ) for i in range(depth)])

        self.num_action_q = num_action_q
        if self.num_action_q > 0:
            self.action_q_vec = nn.Parameter(torch.rand(1, num_action_q, self.latent_embed_dim))

        self.temporal_position_encoding_fn = SinusoidalPositionEncoding(
                                                input_size=latent_embed_dim,
                                                inv_freq_factor=10,
        )

        self.lang_q = nn.Parameter(torch.rand(1, 1, self.latent_embed_dim))
        self.gl_q = nn.Parameter(torch.rand(1, 1, self.latent_embed_dim))
        ## different parameter for each position of instruction. Words embedded using temporal position encoding
        self.instruct_q = nn.Parameter(torch.rand(1, max_instruct_q, self.latent_embed_dim))
        self.img_region_q = nn.Parameter(torch.rand(1, 1, self.latent_embed_dim))
        self.vid_frame_q = nn.Parameter(torch.rand(1, 1, self.latent_embed_dim))
        self.ag_q = nn.Parameter(torch.rand(1, 1, self.latent_embed_dim))
        self.ai_q = nn.Parameter(torch.rand(1, 1, self.latent_embed_dim))

        ### Drops the features after concatenating with classification token
        if query_drop:
            self.query_drop = nn.Dropout(p=drop_rate)
        else:
            self.query_drop = None
        if not query_norm is None:
            self.query_norm = eval(query_norm)(output_size)
        else:
            self.query_norm = None

        ### Initializing Parameter weights separately (?)
        self.apply(self._init_weights)

    def output_shape(self, num_queries):
        return (num_queries, self.latent_embed_dim)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def compute_temporal_mask(self, base_seq_len, seq_len, num_elements, device):
        '''
            base_seq_len: the number of tokens in the layer where queries are generated from
            seq_len: number of tokens in the layer where k, v are generated from
            num_elements: number of elements times to repeat the seq_len
        '''
        original_mask = 1 - (
                torch.triu(torch.ones(base_seq_len, seq_len)) - \
                torch.eye(base_seq_len, seq_len)
        ).to(device)
        mask = original_mask.repeat_interleave(
                num_elements, dim=-1
        ).unsqueeze(dim=0)
        # (1, latent_size_n, N), N = seq_len * num_elements
        return mask

    def get_query_vec(
                self,
                num_input_tokens,
                batch_size,
                action_q_ind=[],
                lang_q_ind=[],
                gl_q_ind=[],
                desc_q_ind=[], ## specifies the positions of words masked.
                instruct_q_ind=[], ## specifies the instruction positions wrt desc_q_ind masked.
                img_region_q_ind=[],
                vid_frame_q_ind=[],
                ai_q_ind=[],
                ag_q_ind=[],
        ):
        '''
            num_input_tokens: number of tokens in input vector. = time
            batch_size: batch_size
            action_q_ind: [bs, num_action_q], position of the action_vectors that needs to be retrieved.
            lang_q_ind: [bs, num_lang_q], position of the masked words in language [original libero annotations].
            gl_q_ind: [bs, num_gl_q], position of the masked words in goal language.
            desc_q_ind: [bs, num_word_q], position of the masked words in each instruction of language detailed instructions
                        NOTE: in the current implementation, it is restricted to one word per language instruction.
            instruct_q_ind: [bs, num_instructions_q], position of all the instructions that have one word masked in the input.
                        NOTE: This is tied to desc_q_ind and should match exactly in dimension
            img_region_q_ind: [bs, num_masked_region_q], position of all the masked regions in the goal image
            vid_frame_q_ind: [bs, num_masked_frames_q], position of all the masked frames in the video demonstration
        '''
        query_meta = {}
        if len(action_q_ind) == 0:
            action_q_ind = batch_size*[[x for x in range(self.num_action_q)]]
        assert len(action_q_ind) == batch_size
        num_action_q = len(action_q_ind[0])
        q_vec = torch.cat([self.action_q_vec[:, action_q_ind[b_ind]] for b_ind in range(batch_size)], dim=0)
        device = q_vec.device

        cross_attn_mask = self.compute_temporal_mask(
                                        base_seq_len=num_action_q,
                                        seq_len=num_input_tokens,
                                        num_elements=1,
                                        device=device
        )
        self_attn_mask = self.compute_temporal_mask(
                                        base_seq_len=num_action_q,
                                        seq_len=num_action_q,
                                        num_elements=1,
                                        device=device
        )
        query_meta = {'action_start_ind': 0,'action_end_ind': num_action_q, 'modalities': ['action']}
        extend_mask_len = 0
        if len(lang_q_ind) > 0:
            query_meta['modalities'].append('lang')
            query_meta['lang_start_ind'] = q_vec.shape[1]
            ## pos_vec: [bs, len(lang_q_ind[0]), E]
            pos_vec = self.temporal_position_encoding_fn(lang_q_ind)
            lang_q_vec = self.lang_q + pos_vec
            ## [bs, q_len, E]
            q_vec = torch.cat((q_vec, lang_q_vec), dim=1)
            extend_mask_len += len(lang_q_ind[0]) ## first position is batch size
            query_meta['lang_end_ind'] = q_vec.shape[1]
        if len(desc_q_ind) > 0:
            assert len(instruct_q_ind) == len(desc_q_ind), "Unequal batch size"
            query_meta['modalities'].append('inst')
            query_meta['inst_start_ind'] = q_vec.shape[1]
            ## pos_vec: [bs, len(lang_q_ind[0]), E]
            pos_vec = self.temporal_position_encoding_fn(desc_q_ind) ## positional embeddings for words
            instruct_q = torch.cat([self.instruct_q[:, instruct_q_ind[b_ind]] for b_ind in range(batch_size)], dim=0)
            desc_q_vec = instruct_q + pos_vec
            q_vec = torch.cat((q_vec, desc_q_vec), dim=1)
            extend_mask_len += len(desc_q_ind[0])
            query_meta['inst_end_ind'] = q_vec.shape[1]
        if len(gl_q_ind) > 0:
            query_meta['modalities'].append('gl')
            query_meta['gl_start_ind'] = q_vec.shape[1]
            ## pos_vec: [bs, len(lang_q_ind[0]), E]
            pos_vec = self.temporal_position_encoding_fn(gl_q_ind)
            gl_q_vec = self.gl_q + pos_vec
            ## [bs, q_len, E]
            q_vec = torch.cat((q_vec, gl_q_vec), dim=1)
            extend_mask_len += len(gl_q_ind[0]) ## first position is batch size
            query_meta['gl_end_ind'] = q_vec.shape[1]
        if len(img_region_q_ind) > 0:
            if isinstance(img_region_q_ind, list): img_region_q_ind = torch.Tensor(img_region_q_ind).long()
            query_meta['modalities'].append('img')
            query_meta['img_start_ind'] = q_vec.shape[1]
            ## [bs, len(img_region_q_ind[0]), E]
            pos_vec = self.temporal_position_encoding_fn(img_region_q_ind)
            img_region_q_vec = self.img_region_q + pos_vec
            ## [bs, q_len, E]
            q_vec = torch.cat((q_vec, img_region_q_vec), dim=1)
            extend_mask_len += len(img_region_q_ind[0]) ## first position is batch size
            query_meta['img_end_ind'] = q_vec.shape[1]
        if len(vid_frame_q_ind) > 0:
            if isinstance(vid_frame_q_ind, list): vid_frame_q_ind = torch.Tensor(vid_frame_q_ind).long()
            query_meta['modalities'].append('vid')
            query_meta['vid_start_ind'] = q_vec.shape[1]
            ## [bs, len(vid_frame_q_ind[0]), E]
            pos_vec = self.temporal_position_encoding_fn(vid_frame_q_ind)
            vid_frame_q_vec = self.vid_frame_q + pos_vec
            ## [bs, q_len, E]
            q_vec = torch.cat((q_vec, vid_frame_q_vec), dim=1)
            extend_mask_len += len(vid_frame_q_ind[0]) ## first position is batch size
            query_meta['vid_end_ind'] = q_vec.shape[1]
        if len(ag_q_ind) > 0:
            if isinstance(ag_q_ind, list): ag_q_ind = torch.Tensor(ag_q_ind).long()
            query_meta['modalities'].append('ag')
            query_meta['ag_start_ind'] = q_vec.shape[1]
            ## [bs, len(ag_q_ind[0]), E]
            pos_vec = self.temporal_position_encoding_fn(ag_q_ind)
            ag_q_vec = self.ag_q + pos_vec
            ## [bs, q_len, E]
            q_vec = torch.cat((q_vec, ag_q_vec), dim=1)
            extend_mask_len += len(ag_q_ind[0]) ## first position is batch size
            query_meta['ag_end_ind'] = q_vec.shape[1]
        if len(ai_q_ind) > 0:
            if isinstance(ai_q_ind, list): ai_q_ind = torch.Tensor(ai_q_ind).long()
            query_meta['modalities'].append('ai')
            query_meta['ai_start_ind'] = q_vec.shape[1]
            ## [bs, len(ai_q_ind[0]), E]
            pos_vec = self.temporal_position_encoding_fn(ai_q_ind)
            ai_q_vec = self.ai_q + pos_vec
            ## [bs, q_len, E]
            q_vec = torch.cat((q_vec, ai_q_vec), dim=1)
            extend_mask_len += len(ai_q_ind[0]) ## first position is batch size
            query_meta['ai_end_ind'] = q_vec.shape[1]

        cross_attn_mask = torch.cat((cross_attn_mask, torch.ones((1, extend_mask_len, num_input_tokens)).to(device)), dim=1) ## dimension 0 is batch size and equal to 1
        self_attn_mask = torch.cat((self_attn_mask, torch.zeros((1, self_attn_mask.shape[-2], extend_mask_len)).to(device)), dim=2)
        self_attn_mask = torch.cat((self_attn_mask, torch.zeros((1, extend_mask_len, self_attn_mask.shape[-1])).to(device)), dim=1)
        self_attn_mask[:,num_action_q:,:] = 1.0

        if not self.query_norm is None:
            q_vec = self.query_norm(q_vec)
        if not self.query_drop is None:
            q_vec = self.query_drop(q_vec) ## Test with query drop

        return q_vec, cross_attn_mask, self_attn_mask, query_meta

    def forward(self, inputs, query_vecs, cross_attn_mask=None, self_attn_mask=None):
        assert query_vecs.shape[2] == self.latent_embed_dim

        x = query_vecs
        for i in range(self.depth):
            if i in self.cross_layers_visual:
                x, _ = self.crossatt_blocks_visual[self.cross_layers_visual.index(i)](x=x, context=inputs, mask=cross_attn_mask)

            x, _ = self.blocks[i](x=x, mask=self_attn_mask)

        return x
