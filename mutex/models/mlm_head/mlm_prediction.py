import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import gelu

class BertPredictionHeadTransform(nn.Module): ## Credits: HuggingFace
    def __init__(self, hidden_act, hidden_size, layer_norm_eps):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        if isinstance(hidden_act, str):
            self.transform_act_fn = eval(hidden_act)
        else:
            self.transform_act_fn = hidden_act
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class MLMHead(nn.Module):
    def __init__(
                    self,
                    hidden_act,
                    layer_norm_eps,
                    vocab_size,
                    hidden_size,
                    mlm_loss_weight,
                    weight=None,
                    anneal_factor=0.1,
                    anneal_every_n_epochs=10
    ):
        super().__init__()
        self.transform = BertPredictionHeadTransform(
                                                hidden_size=hidden_size,
                                                hidden_act=hidden_act, ## bert or clip
                                                layer_norm_eps=layer_norm_eps
        )
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(vocab_size))
        if weight is not None:
            self.decoder.weight = weight
        #self._loss = nn.CrossEntropyLoss(ignore_index=-100)
        self.mlm_loss_weight = mlm_loss_weight
        self.anneal_factor = anneal_factor
        self.anneal_every_n_epochs = anneal_every_n_epochs

    def anneal_weights(self, epoch):
        if (epoch > 0) and (epoch % self.anneal_every_n_epochs == 0):
            self.mlm_loss_weight *= self.anneal_factor

    def forward(self, x):
        x = self.transform(x)
        x = self.decoder(x) + self.bias
        return x

    def loss_fn(self, data, feat):
        assert len(feat.shape) == 3 ## [batch, num_preds, embed_size]
        mlm_indices = data["task_tokens"]["mlm_indices"]
        num_preds = data["task_tokens"]["mlm_indices"].shape[-1]
        gt_ids = data["task_tokens"]["gt_ids"]
        ## Ordering is maintained inside dataset function itself
        ## Order is important. Query vectors for decoder are generated in the order specified in mlm_indices
        #gt_ids = torch.Tensor([[data["task_tokens"]["gt_ids"][b_ind, pred_val] \
        #    for pred_val in  mlm_indices[b_ind]] for b_ind in range(feat.shape[0])]).long().to(feat.device)
        assert num_preds == feat.shape[1]
        logits = self.forward(feat)

        loss = F.cross_entropy(logits.permute(0,2,1), gt_ids, ignore_index=-100)
        return self.mlm_loss_weight * loss, logits

class MIMHead(nn.Module):
    def __init__(
                    self,
                    hidden_act,
                    layer_norm_eps,
                    vocab_size,
                    hidden_size,
                    mlm_loss_weight,
                    weight=None,
                    anneal_factor=0.1,
                    anneal_every_n_epochs=10
    ):
        super().__init__()
        self.transform = BertPredictionHeadTransform(
                                                hidden_size=hidden_size,
                                                hidden_act=hidden_act, ## bert or clip
                                                layer_norm_eps=layer_norm_eps
        )
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(vocab_size))
        if weight is not None:
            self.decoder.weight = weight
        #self._loss = nn.CrossEntropyLoss(ignore_index=-100)
        self.mlm_loss_weight = mlm_loss_weight
        self.anneal_factor = anneal_factor
        self.anneal_every_n_epochs = anneal_every_n_epochs

    def anneal_weights(self, epoch):
        if (epoch > 0) and (epoch % self.anneal_every_n_epochs == 0):
            self.mlm_loss_weight *= self.anneal_factor

    def forward(self, x):
        x = self.transform(x)
        x = self.decoder(x) + self.bias
        return x

    def loss_fn(self, data, feat):
        assert len(feat.shape) == 3 ## [batch, num_preds, embed_size]
        mim_indices = data["mim_indices"]
        num_preds = data["mim_indices"].shape[-1]
        gt_ids = data["gt_inst_ids"]
        ## Ordering is maintained inside dataset function itself
        ## Order is important. Query vectors for decoder are generated in the order specified in mim_indices
        assert num_preds == feat.shape[1]
        logits = self.forward(feat)

        loss = F.cross_entropy(logits.permute(0,2,1), gt_ids, ignore_index=-100)
        return self.mlm_loss_weight * loss, logits

class MGMHead(nn.Module):
    def __init__(
                    self,
                    hidden_act,
                    layer_norm_eps,
                    vocab_size,
                    hidden_size,
                    mlm_loss_weight,
                    weight=None,
                    anneal_factor=0.1,
                    anneal_every_n_epochs=10
    ):
        super().__init__()
        self.transform = BertPredictionHeadTransform(
                                                hidden_size=hidden_size,
                                                hidden_act=hidden_act, ## bert or clip
                                                layer_norm_eps=layer_norm_eps
        )
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(vocab_size))
        if weight is not None:
            self.decoder.weight = weight
        #self._loss = nn.CrossEntropyLoss(ignore_index=-100)
        self.mlm_loss_weight = mlm_loss_weight
        self.anneal_factor = anneal_factor
        self.anneal_every_n_epochs = anneal_every_n_epochs

    def anneal_weights(self, epoch):
        if (epoch > 0) and (epoch % self.anneal_every_n_epochs == 0):
            self.mlm_loss_weight *= self.anneal_factor

    def forward(self, x):
        x = self.transform(x)
        x = self.decoder(x) + self.bias
        return x

    def loss_fn(self, data, feat):
        assert len(feat.shape) == 3 ## [batch, num_preds, embed_size]
        mlm_indices = data["mgm_indices"]
        num_preds = data["mgm_indices"].shape[-1]
        gt_ids = data["gt_gl_ids"]
        ## Ordering is maintained inside dataset function itself
        ## Order is important. Query vectors for decoder are generated in the order specified in mlm_indices
        assert num_preds == feat.shape[1]
        logits = self.forward(feat)

        loss = F.cross_entropy(logits.permute(0,2,1), gt_ids, ignore_index=-100)
        return self.mlm_loss_weight * loss, logits
