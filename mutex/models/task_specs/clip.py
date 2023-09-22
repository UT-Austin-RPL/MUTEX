from typing import Any, Optional, Tuple, Union
import copy
import torch
import numpy as np

from transformers import CLIPVisionModelWithProjection
from transformers.models.clip.modeling_clip import CLIPVisionModelOutput
from mutex.utils import set_requires_grad

class R3Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from r3m import load_r3m
        self.r3m = load_r3m("resnet34")
        self.r3m.eval()
    def create_precomputable_models(self):
        pass
    def pre_compute_feats(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
        ):

        with torch.no_grad():
            encoder_outputs = self.r3m(pixel_values)
        return [encoder_outputs.unsqueeze(1)]

    def post_compute_feats(
            self,
            hidden_embeds,
            attention_mask=None,
    ):
        assert attention_mask is None, "R3M does not support attention_mask"
        return {'image_embeds':hidden_embeds.squeeze(dim=1)}

class CLIPVisionSliced(CLIPVisionModelWithProjection):
    def __init__(self, config):
        super().__init__(config)
        self.pre_compute_vision_model_encoder = torch.nn.Identity() ## to make sure we are calling create_precomputable_models separately
        self.post_compute_vision_model_encoder = torch.nn.Identity()

    def create_precomputable_models(self, layer_ind=10):
        #assert layer_ind <= 12, "Maximum slicing can be done with layer_ind=12 where only the projection layer is computed."
        self.pre_compute_vision_model_encoder = copy.deepcopy(self.vision_model.encoder) ## redundancy? but only done once
        self.post_compute_vision_model_encoder = copy.deepcopy(self.vision_model.encoder)
        self.pre_compute_vision_model_encoder.layers = self.pre_compute_vision_model_encoder.layers[:layer_ind]
        self.post_compute_vision_model_encoder.layers = self.post_compute_vision_model_encoder.layers[layer_ind:]
        del self.vision_model.encoder
        self.train(False)

        set_requires_grad(self.pre_compute_vision_model_encoder, False)
        set_requires_grad(self.vision_model.embeddings, False)
        set_requires_grad(self.vision_model.pre_layrnorm, False)

    def train(
        self,
        mode: bool = True
    ):
        for module in self.children():
            module.train(mode)
        self.training = mode
        if self.pre_compute_vision_model_encoder is not None:
            self.pre_compute_vision_model_encoder.train(False)  ## NO Difference in LayerNorm
        self.vision_model.embeddings.train(False)
        self.vision_model.pre_layrnorm.train(False)

    def remove_precomputed_layers(self):
        '''
            To save GPU memory, cause :))
        '''
        del self.pre_compute_vision_model_encoder
        self.pre_compute_vision_model_encoder = None

    def pre_compute_feats(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ) -> Union[Tuple, CLIPVisionModelOutput]:

        with torch.no_grad():
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            #return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            if pixel_values is None:
                raise ValueError("You have to specify pixel_values")

            hidden_states = self.vision_model.embeddings(pixel_values)
            hidden_states = self.vision_model.pre_layrnorm(hidden_states)

            encoder_outputs = self.pre_compute_vision_model_encoder( ## Replace this
                inputs_embeds=hidden_states,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        return encoder_outputs

    def post_compute_feats(
            self,
            hidden_embeds,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            attention_mask: Optional[bool] = None,
            return_dict=True,
    ) -> Union[Tuple, CLIPVisionModelOutput]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        encoder_outputs = self.post_compute_vision_model_encoder(
            inputs_embeds=hidden_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            attention_mask=attention_mask,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.vision_model.post_layernorm(pooled_output)


        image_embeds = self.visual_projection(pooled_output)

        if not return_dict:
            outputs = (image_embeds, last_hidden_state) + [encoder_outputs.hidden_states, encoder_outputs.attentions]
            return tuple(output for output in outputs if output is not None)

        return CLIPVisionModelOutput(
            image_embeds=image_embeds,
            last_hidden_state=last_hidden_state,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def forward(
        self,
        input,
        **kwargs
    ):
        ## to avoid forward computation usage
        raise NotImplementedError
