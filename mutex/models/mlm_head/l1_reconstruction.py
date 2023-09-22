import torch
import torch.nn as nn

class MAGMHead_L1(nn.Module):
    '''
    this functions calls the L1ReconstructionLoss
    '''
    def __init__(
            self,
            input_size,
            output_size,
            magm_loss_weight=1.0,
            anneal_factor=0.1,
            anneal_every_n_epochs=10
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.magm_loss_weight = magm_loss_weight
        self.anneal_factor = anneal_factor
        self.anneal_every_n_epochs = anneal_every_n_epochs

        # add a torch linear layer for reconstruction
        self.prediction_head = nn.Linear(self.input_size, self.output_size)
        # call _init_weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def anneal_weights(self, epoch):
        # loss weight annealing
        if (epoch > 0) and (epoch % self.anneal_every_n_epochs == 0):
            self.magm_loss_weight *= self.anneal_factor

    def forward(self, x):
        # forward pass
        x = self.prediction_head(x)
        return x

    def loss_fn(self, data, feat):
        assert len(feat.shape) == 3
        bs = feat.size(0)
        # forward pass on feat
        pred = self.forward(feat)
        magm_indices = data['magm_indices']
        gt_ag_spec = data['gt_ag_spec']
        # compute loss
        # use pytorch L1 loss
        loss = nn.L1Loss(reduction='mean')(pred, gt_ag_spec)
        return self.magm_loss_weight * loss

class MAIMHead_L1(nn.Module):
    '''
    this functions calls the L1ReconstructionLoss
    '''
    def __init__(
            self,
            input_size,
            output_size,
            maim_loss_weight=1.0,
            anneal_factor=0.1,
            anneal_every_n_epochs=10
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.maim_loss_weight = maim_loss_weight
        self.anneal_factor = anneal_factor
        self.anneal_every_n_epochs = anneal_every_n_epochs

        # add a torch linear layer for reconstruction
        self.prediction_head = nn.Linear(self.input_size, self.output_size)
        # call _init_weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def anneal_weights(self, epoch):
        # loss weight annealing
        if (epoch > 0) and (epoch % self.anneal_every_n_epochs == 0):
            self.maim_loss_weight *= self.anneal_factor

    def forward(self, x):
        # forward pass
        x = self.prediction_head(x)
        return x

    def loss_fn(self, data, feat):
        assert len(feat.shape) == 3
        bs = feat.size(0)
        # forward pass on feat
        pred = self.forward(feat)
        maim_indices = data['maim_indices']
        gt_ai_spec = data['gt_ai_spec']
        # compute loss
        # use pytorch L1 loss
        loss = nn.L1Loss(reduction='mean')(pred, gt_ai_spec)
        return self.maim_loss_weight * loss

class MRMHead_L1(nn.Module):
    '''
    this functions calls the L1ReconstructionLoss
    '''
    def __init__(
            self,
            input_size,
            output_size,
            mrm_loss_weight=1.0,
            anneal_factor=0.1,
            anneal_every_n_epochs=10
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.mrm_loss_weight = mrm_loss_weight
        self.anneal_factor = anneal_factor
        self.anneal_every_n_epochs = anneal_every_n_epochs

        # add a torch linear layer for reconstruction
        self.prediction_head = nn.Linear(self.input_size, self.output_size)
        # call _init_weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def anneal_weights(self, epoch):
        # loss weight annealing
        if (epoch > 0) and (epoch % self.anneal_every_n_epochs == 0):
            self.mrm_loss_weight *= self.anneal_factor

    def forward(self, x):
        # forward pass
        x = self.prediction_head(x)
        return x

    def loss_fn(self, data, feat):
        assert len(feat.shape) == 3
        bs = feat.size(0)
        # forward pass on feat
        pred = self.forward(feat)
        mrm_indices = data['mrm_indices']
        gt_img_spec = data['gt_img_spec'].reshape(bs, *(data['gt_img_spec'].shape[-2:]))  # [bs, 50, E]
        # compute loss
        # use pytorch L1 loss
        loss = nn.L1Loss(reduction='mean')(pred, gt_img_spec)
        return self.mrm_loss_weight * loss

class MFMHead_L1(nn.Module):
    '''
    this functions calls the L1ReconstructionLoss
    '''
    def __init__(
            self,
            input_size,
            output_size,
            mfm_loss_weight=1.0,
            anneal_factor=0.1,
            anneal_every_n_epochs=10
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.mfm_loss_weight = mfm_loss_weight
        self.anneal_factor = anneal_factor
        self.anneal_every_n_epochs = anneal_every_n_epochs

        # add a torch linear layer for reconstruction
        self.prediction_head = nn.Linear(self.input_size, self.output_size)
        # call _init_weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def anneal_weights(self, epoch):
        # loss weight annealing
        if (epoch > 0) and (epoch % self.anneal_every_n_epochs == 0):
            self.mfm_loss_weight *= self.anneal_factor

    def forward(self, x):
        # forward pass
        x = self.prediction_head(x)
        return x

    def loss_fn(self, data, feat):
        assert len(feat.shape) == 3
        bs = feat.size(0)
        # forward pass on feat
        pred = self.forward(feat)
        mfm_indices = data['mfm_indices']
        gt_vid_spec = data['gt_vid_spec'].reshape(bs, *(data['gt_vid_spec'].shape[-2:]))  # [bs, 50, E]
        # compute loss
        # use pytorch L1 loss
        loss = nn.L1Loss(reduction='mean')(pred, gt_vid_spec)
        return self.mfm_loss_weight * loss
