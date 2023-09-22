import os
import torch
from lifelong.utils import torch_save_model
def torch_save_model(state_dict, model_path, cfg=None, previous_masks=None):
    torch.save({"state_dict": state_dict,
                "cfg": cfg,
                "previous_masks": previous_masks}, model_path)

def main():
    model_path = 'release_exps/mutex/models/ft_LIBERO_100_fx_multitask_model_ep020.pth'
    output_path = 'release_exps/mutex/models/mutex_weights.pth'

    # read the weights
    model = torch.load(model_path)
    state_dict = {}

    # find all weights named as 'language_encoder'
    for key in model['state_dict'].keys():
        if 'language_encoder' in key:
            # rename language_encoder to modality_specific_layers
            new_key = key.replace('language_encoder', 'projection_layers')
            if 'finetune' in key:
                # rename finetune to cross_modal
                new_key = new_key.replace('finetune', 'cross_modal')
                state_dict[new_key] = model['state_dict'][key]
            else:
                state_dict[new_key] = model['state_dict'][key]
        else:
            state_dict[key] = model['state_dict'][key]


    cfg = model['cfg']
    previous_masks = model['previous_masks']
    torch_save_model(state_dict, output_path, cfg, previous_masks)


if __name__ == '__main__':
    main()
