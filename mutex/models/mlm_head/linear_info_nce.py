import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

__all__ = ['InfoNCE', 'info_nce']


class InfoNCE(nn.Module):
    """
    Calculates the InfoNCE loss for self-supervised learning.
    This contrastive loss enforces the embeddings of similar (positive) samples to be close
        and those of different (negative) samples to be distant.
    A query embedding is compared with one positive key and with one or more negative keys.
    References:
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113
    Args:
        temperature: Logits are divided by temperature before calculating the cross entropy.
        reduction: Reduction method applied to the output.
            Value must be one of ['none', 'sum', 'mean'].
            See torch.nn.functional.cross_entropy for more details about each option.
        negative_mode: Determines how the (optional) negative_keys are handled.
            Value must be one of ['paired', 'unpaired'].
            If 'paired', then each query sample is paired with a number of negative keys.
            Comparable to a triplet loss, but with multiple negatives per sample.
            If 'unpaired', then the set of negative keys are all unrelated to any positive key.
    Input shape:
        query: (N, D) Tensor with query samples (e.g. embeddings of the input).
        positive_key: (N, D) Tensor with positive samples (e.g. embeddings of augmented input).
        negative_keys (optional): Tensor with negative samples (e.g. embeddings of other inputs)
            If negative_mode = 'paired', then negative_keys is a (N, M, D) Tensor.
            If negative_mode = 'unpaired', then negative_keys is a (M, D) Tensor.
            If None, then the negative keys for a sample are the positive keys for the other samples.
    Returns:
         Value of the InfoNCE Loss.
     Examples:
        >>> loss = InfoNCE()
        >>> batch_size, num_negative, embedding_size = 32, 48, 128
        >>> query = torch.randn(batch_size, embedding_size)
        >>> positive_key = torch.randn(batch_size, embedding_size)
        >>> negative_keys = torch.randn(num_negative, embedding_size)
        >>> output = loss(query, positive_key, negative_keys)
    """

    def __init__(self, temperature=0.1, reduction='mean', negative_mode='unpaired'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode

    def forward(self, query, positive_key, negative_keys=None):
        return info_nce(query, positive_key, negative_keys,
                        temperature=self.temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode)


def info_nce(query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='unpaired'):
    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        if negative_mode == 'unpaired':
            # Cosine between all query-negative combinations
            negative_logits = query @ transpose(negative_keys)

        elif negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_key)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)


def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]

class MAIMHead(nn.Module):
    """
        MFMHead for Masked Frame Modelling
    """
    def __init__(
            self,
            input_size,
            output_size,
            maim_loss_weight=1.0,
            anneal_factor=0.1,
            anneal_every_n_epochs=10
    ):
        super().__init__()
        self.prediction_head = nn.Linear(input_size, output_size)
        self.info_loss = InfoNCE(negative_mode='paired')  # Credits: https://github.com/RElbers/info-nce-pytorch
        self.maim_loss_weight = maim_loss_weight
        self.anneal_factor = anneal_factor
        self.anneal_every_n_epochs = anneal_every_n_epochs
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def anneal_weights(self, epoch):
        if (epoch > 0) and (epoch % self.anneal_every_n_epochs == 0):
            self.maim_loss_weight *= self.anneal_factor

    def forward(self, x):
        x = self.prediction_head(x)
        return x

    def loss_fn(self, data, feat):
        assert len(feat.shape) == 3  # [batch, num_preds, embed_size]
        pred = self.forward(feat)
        maim_indices = data['maim_indices']
        gt_ai_spec = data['gt_ai_spec']
        not_maim_indices = torch.Tensor([[idx for idx in range(gt_ai_spec.shape[-2]) if not (idx in maim_inds)] for maim_inds in maim_indices]).long()

        # pos[i,j,k] = gt_vid_spec[i, mfm_indices[i,j,k], k]
        pos = torch.gather(gt_ai_spec, dim=1, index=maim_indices.unsqueeze(dim=-1).repeat(1, 1, gt_ai_spec.shape[-1]))

        B, num_query = pos.shape[:2]
        pred = pred.reshape(B*num_query, -1)
        pos = pos.reshape(B*num_query, -1)
        not_maim_indices = not_maim_indices.reshape(B*num_query, -1).to(gt_ai_spec.device)
        neg = gt_ai_spec.repeat_interleave(num_query, dim=0)
        neg = torch.gather(neg, dim=1, index=not_maim_indices.unsqueeze(dim=-1).repeat(1, 1, neg.shape[-1]))

        loss = self.info_loss(pred, pos, neg)
        return self.maim_loss_weight * loss

class MAGMHead(nn.Module):
    """
        MFMHead for Masked Frame Modelling
    """
    def __init__(
            self,
            input_size,
            output_size,
            magm_loss_weight=1.0,
            anneal_factor=0.1,
            anneal_every_n_epochs=10
    ):
        super().__init__()
        self.prediction_head = nn.Linear(input_size, output_size)
        self.info_loss = InfoNCE(negative_mode='paired')  # Credits: https://github.com/RElbers/info-nce-pytorch
        self.magm_loss_weight = magm_loss_weight
        self.anneal_factor = anneal_factor
        self.anneal_every_n_epochs = anneal_every_n_epochs
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def anneal_weights(self, epoch):
        if (epoch > 0) and (epoch % self.anneal_every_n_epochs == 0):
            self.magm_loss_weight *= self.anneal_factor

    def forward(self, x):
        x = self.prediction_head(x)
        return x

    def loss_fn(self, data, feat):
        assert len(feat.shape) == 3  # [batch, num_preds, embed_size]
        pred = self.forward(feat)
        magm_indices = data['magm_indices']
        gt_ag_spec = data['gt_ag_spec']
        not_magm_indices = torch.Tensor([[idx for idx in range(gt_ag_spec.shape[-2]) if not (idx in magm_inds)] for magm_inds in magm_indices]).long()

        # pos[i,j,k] = gt_vid_spec[i, mfm_indices[i,j,k], k]
        pos = torch.gather(gt_ag_spec, dim=1, index=magm_indices.unsqueeze(dim=-1).repeat(1, 1, gt_ag_spec.shape[-1]))

        B, num_query = pos.shape[:2]
        pred = pred.reshape(B*num_query, -1)
        pos = pos.reshape(B*num_query, -1)
        not_magm_indices = not_magm_indices.reshape(B*num_query, -1).to(gt_ag_spec.device)
        neg = gt_ag_spec.repeat_interleave(num_query, dim=0)
        neg = torch.gather(neg, dim=1, index=not_magm_indices.unsqueeze(dim=-1).repeat(1, 1, neg.shape[-1]))

        loss = self.info_loss(pred, pos, neg)
        return self.magm_loss_weight * loss

class MFMHead(nn.Module):
    """
        MFMHead for Masked Frame Modelling
    """
    def __init__(
            self,
            input_size,
            output_size,
            mfm_loss_weight=1.0,
            anneal_factor=0.1,
            anneal_every_n_epochs=10
    ):
        super().__init__()
        self.prediction_head = nn.Linear(input_size, output_size)
        self.info_loss = InfoNCE(negative_mode='paired')  # Credits: https://github.com/RElbers/info-nce-pytorch
        self.mfm_loss_weight = mfm_loss_weight
        self.anneal_factor = anneal_factor
        self.anneal_every_n_epochs = anneal_every_n_epochs
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def anneal_weights(self, epoch):
        if (epoch > 0) and (epoch % self.anneal_every_n_epochs == 0):
            self.mfm_loss_weight *= self.anneal_factor

    def forward(self, x):
        x = self.prediction_head(x)
        return x

    def loss_fn(self, data, feat):
        assert len(feat.shape) == 3  # [batch, num_preds, embed_size]
        pred = self.forward(feat)
        mfm_indices = data['mfm_indices']
        gt_vid_spec = data['gt_vid_spec']
        not_mfm_indices = torch.Tensor(
                [[[idx for idx in range(gt_vid_spec.shape[1]) if not (idx == mfm_id)] for mfm_id in mfm_inds] for mfm_inds in mfm_indices]
        ).long()

        # pos[i,j,k] = gt_vid_spec[i, mfm_indices[i,j,k], k]
        pos = torch.gather(gt_vid_spec, dim=1, index=mfm_indices.unsqueeze(dim=-1).repeat(1, 1, gt_vid_spec.shape[-1]))

        B, num_query = pos.shape[:2]
        pred = pred.reshape(B*num_query, -1)
        pos = pos.reshape(B*num_query, -1)
        not_mfm_indices = not_mfm_indices.reshape(B*num_query, -1).to(gt_vid_spec.device)
        neg = gt_vid_spec.repeat_interleave(num_query, dim=0)
        neg = torch.gather(neg, dim=1, index=not_mfm_indices.unsqueeze(dim=-1).repeat(1, 1, neg.shape[-1]))

        loss = self.info_loss(pred, pos, neg)
        return self.mfm_loss_weight * loss

class MRMHead(nn.Module):
    def __init__(
            self,
            input_size,
            output_size,
            mrm_loss_weight=1.0,
            anneal_factor=0.1,
            anneal_every_n_epochs=10
    ):
        super().__init__()
        self.prediction_head = nn.Linear(input_size, output_size)
        # Credits: https://github.com/RElbers/info-nce-pytorch
        self.info_loss = InfoNCE(negative_mode='paired')
        self.mrm_loss_weight = mrm_loss_weight
        self.anneal_factor = anneal_factor
        self.anneal_every_n_epochs = anneal_every_n_epochs
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def anneal_weights(self, epoch):
        if (epoch > 0) and (epoch % self.anneal_every_n_epochs == 0):
            self.mrm_loss_weight *= self.anneal_factor

    def forward(self, x):
        x = self.prediction_head(x)
        return x

    def loss_fn(self, data, feat):
        assert len(feat.shape) == 3  # [batch, num_preds, embed_size]
        bs = feat.size(0)
        pred = self.forward(feat)  # [bs, num_region_ind, E]
        mrm_indices = data['mrm_indices']  # [bs, num_region_ind]
        gt_img_spec = data['gt_img_spec'].reshape(bs, *(data['gt_img_spec'].shape[-2:]))  # [bs, 50, E]

        # use sets instead of this loops???
        # TODO: optimize this, why did I use for loop? lazy me
        # [bs, num_query, 50-num_masked_regions]
        not_mrm_indices = torch.Tensor(
                [[[idx for idx in range(gt_img_spec.shape[-2]) if not (idx == mrm_id)] for mrm_id in mrm_inds] for mrm_inds in mrm_indices]
        ).long()

        # pos[i,j,k] = gt_img_spec[i, mrm_indices[i,j,k], k]
        pos = torch.gather(gt_img_spec, dim=1, index=mrm_indices.unsqueeze(dim=-1).repeat(1, 1, gt_img_spec.shape[-1]))

        B, num_query = pos.shape[:2]
        pred = pred.reshape(B*num_query, -1)
        pos = pos.reshape(B*num_query, -1)
        not_mrm_indices = not_mrm_indices.reshape(B*num_query, -1).to(gt_img_spec.device)
        neg = gt_img_spec.repeat_interleave(num_query, dim=0)
        neg = torch.gather(neg, dim=1, index=not_mrm_indices.unsqueeze(dim=-1).repeat(1, 1, neg.shape[-1]))

        loss = self.info_loss(pred, pos, neg)
        return self.mrm_loss_weight * loss
