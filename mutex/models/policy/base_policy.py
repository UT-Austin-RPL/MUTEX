import robomimic.utils.tensor_utils as TensorUtils
import torch.nn as nn

from mutex.models.data_augmentation import *


class BasePolicy(nn.Module):
    def __init__(self, cfg, shape_meta):
        super().__init__()
        self.cfg = cfg
        self.shape_meta = shape_meta

        policy_cfg = cfg.policy
        # add image augmentation
        color_aug = eval(policy_cfg.color_aug.network)(
                **policy_cfg.color_aug.network_kwargs)
        policy_cfg.translation_aug.network_kwargs["input_shape"] = \
                shape_meta["all_shapes"]["agentview_rgb"]
        translation_aug = eval(policy_cfg.translation_aug.network)(
                **policy_cfg.translation_aug.network_kwargs)
        self.img_aug = DataAugGroup((color_aug, translation_aug))

        # for visualization
        self.handles = []
        self.gradients = None
        self.activations = None
        self.reshape_transform = None

    def forward(self, data):
        raise NotImplementedError

    def get_action(self, data):
        raise NotImplementedError

    def get_action_and_attention(self, data):
        raise NotImplementedError

    def register_hook_for_attention(self):
        raise NotImplementedError

    def release_handle(self):
        for handle in self.handles:
            handle.remove()

    def _get_img_tuple(self, data):
        img_tuple = tuple([
            data["obs"][img_name] for img_name in self.image_encoders.keys()
        ])
        return img_tuple

    def _get_aug_output_dict(self, out):
        img_dict = {
            img_name: out[idx] for idx, img_name in enumerate(self.image_encoders.keys())
        }
        return img_dict

    def process_input(self, data, train_mode=True):
        if train_mode: # apply augmentation
            if self.cfg.train.use_augmentation:
                img_tuple = self._get_img_tuple(data)
                aug_out = self._get_aug_output_dict(self.img_aug(img_tuple))
                for img_name in self.image_encoders.keys():
                    data["obs"][img_name] = aug_out[img_name]
            return data
        else:
            data = TensorUtils.recursive_dict_list_tuple_apply(data, {
                torch.Tensor: lambda x: x.unsqueeze(dim=1) # add time dimension
            })
            data["task_emb"] = data["task_emb"].squeeze(1)
        return data

    def get_loss(self, data, reduction='mean'):
        log_info = {}
        data = self.process_input(data, train_mode=True)
        dist = self.forward(data)
        loss = self.policy_head.loss_fn(dist, data["actions"], reduction)
        log_info['bc_loss'] = loss.item()
        log_info['total_loss'] = loss.item()
        return loss, log_info

    def reset(self):
        pass

    def save_activation(self, module, input, output):
        activation = output
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations = activation.cpu().detach()

    def save_gradient(self, module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            # You can only register hooks on tensor requires grad.
            return
        # Gradients are computed in reverse order
        def _store_grad(grad):
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)
            self.gradients = grad.cpu().detach()
        output.register_hook(_store_grad)
