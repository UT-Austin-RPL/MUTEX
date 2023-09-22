import math
import numpy as np
from torch import nn
import torch
import torchvision
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


# Simple components for building transformer model

class Norm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, head_output_size=64, dropout=0., attn_dropout=0.):
        super().__init__()

        self.num_heads = num_heads
        # \sqrt{d_{k}}
        self.att_scale = head_output_size ** (-0.5)
        self.qkv = nn.Linear(dim,
                             num_heads*head_output_size*3,
                             bias=False)

        # We need to combine the output from all heads
        self.output_layer = nn.Sequential(
                nn.Linear(num_heads * head_output_size, dim),
                nn.Dropout(dropout)
        )
        self.attn_drop = nn.Dropout(attn_dropout)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, -1)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (qkv[0], qkv[1], qkv[2])

        # q.dot(k.transpose)
        attn = (q @ k.transpose(-2, -1)) * self.att_scale
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

        # (..., num_heads, seq_len, head_output_size)
        out = rearrange(torch.matmul(attn, v), 'b h n d -> b n (h d)')
        return self.output_layer(out)

class CrossAttention(nn.Module):
    def __init__(self, dim, kv_dim, num_heads=8, head_output_size=64, dropout=0., attn_dropout=0.):
        super().__init__()

        self.num_heads = num_heads
        # \sqrt{d_{k}}
        self.att_scale = head_output_size ** (-0.5)
        self.kv_dim = kv_dim
        self.kv = nn.Linear(kv_dim,
                             num_heads*head_output_size*2,
                             bias=False)
        self.q = nn.Linear(dim,
                             num_heads*head_output_size*1,
                             bias=False)

        # We need to combine the output from all heads
        self.output_layer = nn.Sequential(
                nn.Linear(num_heads * head_output_size, dim),
                nn.Dropout(dropout)
        )
        self.attn_drop = nn.Dropout(attn_dropout)

    def forward(self, x, context, mask=None):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q = q[0]
        N_kv = context.shape[1]
        kv = (
            self.kv(context)
            .reshape(B, N_kv, 2, self.num_heads, -1)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv[0], kv[1]

        # q.dot(k.transpose)
        attn = (q @ k.transpose(-2, -1)) * self.att_scale
        if mask is not None:
            mask = mask.bool()
            if len(mask.shape) == 2: # (B, N)
                attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))
            elif len(mask.shape) == 3 and mask.shape[0] == 1: # (1, N, N)
                attn = attn.masked_fill(~mask[None, :, :, :], float("-inf"))
            elif len(mask.shape) == 3 and mask.shape[0] == B:
                attn = attn.masked_fill(~mask[:, None, :, :], float("-inf"))
            else:
                raise Exception("mask shape is not correct for attention")
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # (..., num_heads, seq_len, head_output_size)
        out = rearrange(torch.matmul(attn, v), 'b h n d -> b n (h d)')
        return self.output_layer(out)



class TransformerFeedForwardNN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        # Remember the residual connection
        layers = [nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, dim),
                nn.Dropout(dropout)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


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

    def output_shape(self, input_shape):
        return input_shape

    def output_size(self, input_size):
        return input_size
