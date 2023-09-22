import copy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import get_original_cwd, to_absolute_path

from transformers import AutoModel, AutoTokenizer
from transformers import CLIPTextModelWithProjection, CLIPVisionModelWithProjection
from transformers import T5Tokenizer, T5ForConditionalGeneration
import robomimic.utils.tensor_utils as TensorUtils

from mutex.utils import set_requires_grad
from mutex.models.data_augmentation import *
from mutex.models.image_encoder import *
from mutex.models.projection_layer import *
from mutex.models.decoder import TransformerCrossDecoder
from mutex.models.policy.base_policy import BasePolicy
from mutex.models.policy_head import *
from mutex.models.transformer import *
from mutex.models.mlm_head import MFMHead, MRMHead, MIMHead, MGMHead, \
        MAIMHead, MAGMHead, MRMHead_L1, MFMHead_L1, MAIMHead_L1, MAGMHead_L1
from mutex.models.task_specs import CLIPVisionSliced

###############################################################################
#
# A model handling extra input modalities besides images at time t.
#
###############################################################################

class ExtraModalityTokens(nn.Module):
    def __init__(self,
                 use_joint=False,
                 use_gripper=False,
                 gripper_dim=2,
                 use_ee=False,
                 extra_num_layers=0,
                 extra_hidden_size=64,
                 extra_embedding_size=32):
        """
        This is a class that maps all extra modality inputs into tokens of the same size
        """
        super().__init__()
        self.use_joint = use_joint
        self.use_gripper = use_gripper
        self.use_ee = use_ee
        self.extra_embedding_size = extra_embedding_size

        joint_states_dim = 7
        gripper_states_dim = gripper_dim
        ee_dim = 3

        self.num_extra = int(use_joint) + int(use_gripper) + int(use_ee)

        extra_low_level_feature_dim = int(use_joint) * joint_states_dim + \
                int(use_gripper) * gripper_states_dim + \
                int(use_ee) * ee_dim

        assert extra_low_level_feature_dim > 0, "[error] no extra information"

        self.extra_encoders = nn.ModuleDict({})

        for (proprio_dim, use_modality, modality_name) in [
                (joint_states_dim, self.use_joint, "joint_states"),
                (gripper_states_dim, self.use_gripper, "gripper_states"),
                (ee_dim, self.use_ee, "ee_states")]:

            if use_modality:
                assert proprio_dim > 0 # we indeed have extra information
                if extra_num_layers > 0:
                    layers = [
                        nn.Linear(proprio_dim, extra_hidden_size)
                    ]
                    for i in range(1, extra_num_layers):
                        layers += [
                            nn.Linear(extra_hidden_size, extra_hidden_size),
                            nn.ReLU(inplace=True)
                        ]
                    layers += [nn.Linear(extra_hidden_size, extra_embedding_size)]
                else:
                    layers = [
                        nn.Linear(proprio_dim, extra_embedding_size)
                    ]

                self.extra_encoders[modality_name] = nn.ModuleDict({"encoder": nn.Sequential(*layers)})

        self.encoders = nn.ModuleList([
            x["encoder"] for x in self.extra_encoders.values()])

    def forward(self, obs_dict):
        """
        obs_dict: {
            (optional) joint_stats: (B, T, 7),
            (optional) gripper_states: (B, T, 2),
            (optional) ee: (B, T, 3)
        }
        map above to a latent vector of shape (B, T, H)
        """
        tensor_list = []

        for (use_modality, modality_name) in [
                (self.use_joint, "joint_states"),
                (self.use_gripper, "gripper_states"),
                (self.use_ee, "ee_states")]:

            if use_modality:
                tensor_list.append(self.extra_encoders[modality_name]["encoder"](obs_dict[modality_name]))

        x = torch.stack(tensor_list, dim=-2)
        return x


class BCMutexPolicy(BasePolicy):
    """
    Input: (o_{t-H}, ... , o_t)
    Output: a_t or distribution of a_t
    """
    def __init__(self,
                 cfg,
                 shape_meta):
        super().__init__(cfg, shape_meta)
        policy_cfg = cfg.policy

        ### 1. encode image
        embed_size = policy_cfg.embed_size
        transformer_input_sizes = []
        self.image_encoders = nn.ModuleDict({})
        for name in shape_meta["all_shapes"].keys():
            if "rgb" in name or "depth" in name:
                kwargs = policy_cfg.image_encoder.network_kwargs
                kwargs.input_shape = shape_meta["all_shapes"][name]
                kwargs.output_size = embed_size
                kwargs.language_dim = embed_size
                self.image_encoders[name] = nn.ModuleDict({
                    "encoder": eval(policy_cfg.image_encoder.network)(**kwargs)
                })

        self.encoders = nn.ModuleList([x["encoder"] for x in self.image_encoders.values()])
        # stop the gradients of the video representations. Only used during cross-modal matching.
        self.sg_gt_rep = policy_cfg.sg_gt_rep
        # add language similar to RT1
        self.use_film = policy_cfg.use_film
        # ShareMLP Block between modalities.
        self.add_extra_mlp = policy_cfg.add_extra_mlp
        # Adds representation matching loss
        self.add_rep_loss = policy_cfg.add_rep_loss
        # Representation matching loss co-efficient
        self.rep_loss_coef = policy_cfg.rep_loss_coef

        # define projection_layers Note: SharedMLP is only used when add_extra_mlp=True.
        policy_cfg.projection_layer.network_kwargs.output_size = policy_cfg.projection_layer.network_kwargs.input_size
        self.projection_layers = eval(policy_cfg.projection_layer.network)(
                **policy_cfg.projection_layer.network_kwargs)

        ### encode extra information (e.g. gripper, joint_state)
        real_robot = False
        self.extra_encoder = ExtraModalityTokens(
                 use_joint=cfg.data.use_joint,
                 use_gripper=cfg.data.use_gripper,
                 use_ee=cfg.data.use_ee,
                 gripper_dim = 1 if real_robot else 2,
                 extra_num_layers=policy_cfg.extra_num_layers,
                 extra_hidden_size=policy_cfg.extra_hidden_size,
                 extra_embedding_size=embed_size)

        ### define temporal encoding for observation tokens
        policy_cfg.temporal_position_encoding.network_kwargs.input_size = embed_size
        self.temporal_position_encoding_fn = eval(
                policy_cfg.temporal_position_encoding.network
        )(**policy_cfg.temporal_position_encoding.network_kwargs)

        ### define Policy Encoder
        self.temporal_transformer = TransformerCrossEncoder(
                 input_size=embed_size,
                 context_size=policy_cfg.projection_layer.network_kwargs.output_size,
                 num_layers=policy_cfg.transformer_num_layers,
                 num_heads=policy_cfg.transformer_num_heads,
                 head_output_size=policy_cfg.transformer_head_output_size,
                 mlp_hidden_size=policy_cfg.transformer_mlp_hidden_size,
                 dropout=policy_cfg.transformer_dropout,
                 attn_dropout=policy_cfg.transformer_attn_dropout,
                 cross_attn_ind=policy_cfg.transformer_cross_attn_ind)

        ### define Policy Decoder
        policy_cfg.decoder.network_kwargs.input_size = embed_size
        policy_cfg.decoder.network_kwargs.output_size = embed_size
        self.decoder = eval(policy_cfg.decoder.network)(
                **policy_cfg.decoder.network_kwargs)

        ### define Policy Head
        policy_head_kwargs = policy_cfg.policy_head.network_kwargs
        policy_head_kwargs.input_size = policy_cfg.decoder.network_kwargs.output_size
        policy_head_kwargs.output_size = shape_meta["ac_dim"]

        self.policy_head = eval(policy_cfg.policy_head.network)(
                **policy_cfg.policy_head.loss_kwargs,
                **policy_cfg.policy_head.network_kwargs)

        self.task_spec_modalities = cfg.policy.task_spec_modalities.split('_')
        ### Setting up language feature extractor
        if cfg.lang_embedding_format == "clip":
            self.language_emb_model = CLIPTextModelWithProjection.from_pretrained(
                                                                cfg.lang_tokenizer,
                                                                cache_dir=to_absolute_path("./clip")).eval()
            self.lang_output_key = "text_embeds"
        else:
            raise NotImplementedError
        ### Setting up visual feature extractor for image goals
        if cfg.visual_embedding_format == "clip":
            self.visual_emb_model = CLIPVisionSliced.from_pretrained(
                                                    cfg.tokenizer,
                                                    cache_dir=to_absolute_path("./clip"))
            self.visual_emb_model.create_precomputable_models(layer_ind=cfg.policy.slice_model_ind)
            self.visual_emb_model.remove_precomputed_layers()
        else:
            raise NotImplementedError
        set_requires_grad(self.language_emb_model, mode=False)
        set_requires_grad(self.visual_emb_model, mode=False)

        ### Masked Modeling Heads
        self.add_mfm = cfg.policy.add_mfm
        self.add_mrm = cfg.policy.add_mrm
        self.add_mim = cfg.policy.add_mim
        self.add_mgm = cfg.policy.add_mgm
        self.add_magm = cfg.policy.add_magm
        self.add_maim = cfg.policy.add_maim
        if self.add_mfm:
            cfg.policy.mfm_head.network_kwargs.input_size = cfg.policy.embed_size
            self.mfm_head = eval(cfg.policy.mfm_head.network)(**cfg.policy.mfm_head.network_kwargs)
        if self.add_mrm:
            cfg.policy.mrm_head.network_kwargs.input_size = cfg.policy.embed_size
            self.mrm_head = eval(cfg.policy.mrm_head.network)(**cfg.policy.mrm_head.network_kwargs)
        if self.add_mim:
            cfg.policy.mim_head.network_kwargs.hidden_size = cfg.policy.embed_size
            kwargs = cfg.policy.mim_head.network_kwargs
            text_config = self.language_emb_model.config
            self.mim_head = eval(cfg.policy.mim_head.network)(
                                                    hidden_act="gelu",
                                                    layer_norm_eps=text_config.layer_norm_eps if hasattr(text_config, "layer_norm_eps") else text_config.layer_norm_epsilon,
                                                    vocab_size=text_config.vocab_size,
                                                    **kwargs)
        if self.add_mgm:
            cfg.policy.mgm_head.network_kwargs.hidden_size = cfg.policy.embed_size
            kwargs = cfg.policy.mgm_head.network_kwargs
            text_config = self.language_emb_model.config
            self.mgm_head = eval(cfg.policy.mgm_head.network)(
                                                    hidden_act="gelu",
                                                    layer_norm_eps=text_config.layer_norm_eps if hasattr(text_config, "layer_norm_eps") else text_config.layer_norm_epsilon,
                                                    vocab_size=text_config.vocab_size,
                                                    **kwargs)
        if self.add_magm:
            cfg.policy.magm_head.network_kwargs.input_size = cfg.policy.embed_size
            self.magm_head = eval(cfg.policy.magm_head.network)(**cfg.policy.magm_head.network_kwargs)
        if self.add_maim:
            cfg.policy.maim_head.network_kwargs.input_size = cfg.policy.embed_size
            self.maim_head = eval(cfg.policy.maim_head.network)(**cfg.policy.maim_head.network_kwargs)

        self.cross_modal_matching = False
        ## Debugging model
        self.debug = cfg.train.debug
        if self.debug:
            self.tz = AutoTokenizer.from_pretrained(cfg.lang_tokenizer, cache_dir=to_absolute_path("./bert"))

        self.log_info = {}
        # For evaluation to reuse
        self.latent_queue = []
        self.max_seq_len = policy_cfg.transformer_max_seq_len

    def train(self, mode=True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)

        self.language_emb_model.train(False) ## Always at eval
        if not self.visual_emb_model is None:
            self.visual_emb_model.train(False) ## Always at eval

    def anneal_weights(self, epoch):
        if self.add_mrm:
            self.mrm_head.anneal_weights(epoch)
        if self.add_mfm:
            self.mfm_head.anneal_weights(epoch)
        if self.add_mim:
            self.mim_head.anneal_weights(epoch)
        if self.add_mgm:
            self.mgm_head.anneal_weights(epoch)
        if self.add_maim:
            self.maim_head.anneal_weights(epoch)
        if self.add_magm:
            self.magm_head.anneal_weights(epoch)

    def get_task_embs(self, data, modalities=None):
        if modalities is None:
            modalities = self.task_spec_modalities ## IF not passed explicitly, pick all modalities
        num_modalities = len(self.task_spec_modalities)
        if self.training:
            num_modalities = random.randint(1, len(modalities))
            selected_modalities = random.sample(modalities, num_modalities)
            # cross_modal_matching is set to True only during cross modal matching
            while self.cross_modal_matching and (num_modalities == 1) and (selected_modalities[0] == 'vid'):
                num_modalities = random.randint(1, len(modalities))
                selected_modalities = random.sample(modalities, num_modalities)
                if (num_modalities > 1) or (selected_modalities[0] != 'vid'):
                    break
            modalities = selected_modalities

        inst_emb, gl_emb, img_emb, vid_emb, ag_emb, ai_emb = None, None, None, None, None, None
        inst_emb_mask, gl_emb_mask, img_emb_mask, vid_emb_mask, ag_emb_mask, ai_emb_mask = None, None, None, None, None, None
        if 'inst' in modalities:
            if 'inst_emb' in data:
                inst_emb = data['inst_emb']
            else:
                input_ids = data["inst_tokens"]["input_ids"]
                attention_mask = data["inst_tokens"]["attention_mask"]
                bs, t, e = input_ids.shape
                inst_emb = self.language_emb_model(
                                            input_ids=input_ids.reshape(bs*t, -1),
                                            attention_mask=attention_mask.reshape(bs*t, -1) if num_modalities > 1 else None,
                )[self.lang_output_key]
                if 'hidden_state' in self.lang_output_key:  # if we are taking hidden states, we need to take the mean of all the hidden tokens
                    inst_emb = torch.mean(inst_emb, dim=-2)
                inst_emb = inst_emb.reshape(bs, t, inst_emb.size(-1))


            inst_emb = self.projection_layers.add_temporal_token(inst_emb)
            inst_emb_mask = data['inst_emb_mask'].to(inst_emb.device)
        if 'gl' in modalities:
            if 'gl_emb' in data:
                gl_emb = data['gl_emb'] ## batch size should be handled in the other part
            else:
                gl_emb = self.language_emb_model(
                                        input_ids=data["gl_tokens"]["input_ids"],
                                        attention_mask=data["gl_tokens"]["attention_mask"] if num_modalities > 1 else None,
                )[self.lang_output_key].unsqueeze(dim=1) ## [B, 1, E]
                if 'hidden_state' in self.lang_output_key:  # if we are taking hidden states, we need to take the mean of all the hidden tokens
                    gl_emb = torch.mean(gl_emb, dim=-2)

            gl_emb_mask = torch.ones(gl_emb.shape[:-1]).to(gl_emb.device)
        if 'img' in modalities:
            img_spec = data['img_spec']
            img_spec_mask = data['img_spec_mask']
            B, T = img_spec.shape[:2]
            img_spec = img_spec.reshape(B*T, img_spec.shape[2], img_spec.shape[3])#.to(self.device)
            ## we pass the attention mask since that feature shouldn't be used to compute the final feature
            img_emb = self.visual_emb_model.post_compute_feats(
                        img_spec,
                        attention_mask=img_spec_mask if num_modalities > 1 else None
            )['image_embeds'].reshape(B, T, -1) ## [B, 1, E]
            ## this feature is safe to use and hence, keep the mask as all 1
            img_emb_mask = torch.ones(img_emb.shape[:-1]).to(img_spec.device)
        if 'vid' in modalities:
            vid_emb = data['vid_spec']
            vid_emb_mask = data['vid_spec_mask'] if num_modalities > 1 else torch.ones(data['vid_spec_mask'].shape).to(vid_emb.device) ## don't mask if only one modality
            vid_emb = self.projection_layers.add_temporal_token(vid_emb) ## It is called language encoder for backward compatibility
        if 'ag' in modalities:
            ag_emb = data['ag_task_spec']
            ag_emb_mask = data['ag_task_spec_mask'] \
                    if num_modalities > 1 \
                    else torch.ones(data['ag_task_spec_mask'].shape).to(ag_emb.device)
        if 'ai' in modalities:
            ai_emb = data['ai_task_spec']
            ai_emb_mask = data['ai_task_spec_mask'] \
                    if num_modalities > 1 \
                    else torch.ones(data['ai_task_spec_mask'].shape).to(ai_emb.device)
        assert not (img_emb is None and vid_emb is None and inst_emb is None and gl_emb is None and ai_emb is None and ag_emb is None)

        emb, emb_mask, gt_rep = self.projection_layers.aggregate_task_embs(
                                emb_dict={'inst_emb': inst_emb, 'gl_emb': gl_emb, \
                                        'img_emb': img_emb, 'vid_emb': vid_emb, 'ag_emb': ag_emb, 'ai_emb': ai_emb},
                                emb_mask_dict={'inst_emb_mask': inst_emb_mask, 'gl_emb_mask': gl_emb_mask, \
                                        'img_emb_mask': img_emb_mask, 'vid_emb_mask': vid_emb_mask, 'ag_emb_mask': ag_emb_mask, 'ai_emb_mask': ai_emb_mask},
                                return_gt_rep=self.add_rep_loss,
                                sg_gt_rep=self.sg_gt_rep)
        rep_dict = {'task_emb': emb.clone(), 'gt_rep': gt_rep.clone() if gt_rep is not None else None}

        # note sg_gt_rep is not used in the current if extra_mlp is True, these layers will be tuned for gt_rep
        if self.add_extra_mlp:
            bs, t, e = emb.shape
            emb = self.projection_layers(emb.reshape(bs*t, e)).reshape(bs, t, -1)
        return emb, emb_mask, modalities, rep_dict

    def temporal_encode(self, x, context_tokens, context_tokens_mask=None):
        pos_emb = self.temporal_position_encoding_fn(x)
        x = x + pos_emb.unsqueeze(1) # (B, T, num_modality, E)
        sh = x.shape
        self.temporal_transformer.compute_mask(x.shape, context_shape=context_tokens.shape, context_mask=context_tokens_mask)

        x = TensorUtils.join_dimensions(x, 1, 2) # (B, T*num_modality, E)
        x = self.temporal_transformer(x, context=context_tokens)
        x = x.reshape(*sh)
        return x[:,:,0] # (B, T, E)

    def spatial_encode(self, data):
        # 1. encode extra
        extra = self.extra_encoder(data["obs"]) # (B, T, num_extra, E)

        # 2. encode language, treat it as action token
        B, T = extra.shape[:2]
        encoded = [extra]

        # 1. encode image
        for img_name in self.image_encoders.keys():
            x = data["obs"][img_name]
            B, T, C, H, W = x.shape

            img_encoded = self.image_encoders[img_name]["encoder"](
                    x.reshape(B*T, C, H, W),
                    langs=None, ## [B,T,1,E] --> [B*T, E]
            ).view(B, T, 1, -1)
            encoded.append(img_encoded)
        encoded = torch.cat(encoded, -2) # (B, T, num_modalities, E)
        return encoded

    def forward(self, data, reduction='mean'):
        data = self.process_input(data, train_mode=True) ## Even for validation keep train_mode=True
        data["task_emb"], data["task_emb_mask"], inp_modalities, rep_dict = self.get_task_embs(data)
        x = self.spatial_encode(data)
        x = self.temporal_encode(x, context_tokens=data['task_emb'], context_tokens_mask=data['task_emb_mask'])

        add_mrm = self.add_mrm and (len(inp_modalities) > 1) and ('img' in inp_modalities)
        add_mfm = self.add_mfm and (len(inp_modalities) > 1) and ('vid' in inp_modalities)
        add_mim = self.add_mim and (len(inp_modalities) > 1) and ('inst' in inp_modalities)
        add_mgm = self.add_mgm and (len(inp_modalities) > 1) and ('gl' in inp_modalities)
        add_maim = self.add_maim and (len(inp_modalities) > 1) and ('ai' in inp_modalities)
        add_magm = self.add_magm and (len(inp_modalities) > 1) and ('ag' in inp_modalities)
        q_x, cross_attn_mask, self_attn_mask, query_meta = self.decoder.get_query_vec(
                    num_input_tokens=x.shape[1],
                    batch_size=x.shape[0],
                    lang_q_ind=[],
                    vid_frame_q_ind=data['mfm_indices'] if add_mfm else [],
                    img_region_q_ind=data['mrm_indices'] if add_mrm else [],
                    instruct_q_ind=data['mim_indices'] if add_mim else [],
                    desc_q_ind=data['desc_indices'] if add_mim else [],
                    gl_q_ind=data['mgm_indices'] if add_mgm else [],
                    ai_q_ind=data['maim_indices'] if add_maim else [],
                    ag_q_ind=data['magm_indices'] if add_magm else [],
                    action_q_ind=[])
        x = self.decoder(inputs=x, query_vecs=q_x, cross_attn_mask=cross_attn_mask, self_attn_mask=self_attn_mask)

        dist = self.policy_head(x[:,query_meta['action_start_ind']:query_meta['action_end_ind'],:])

        loss = 0.0

        loss = 0.0
        bc_loss = self.policy_head.loss_fn(dist, data["actions"], reduction)
        loss += bc_loss

        self.log_info['bc_loss'] = bc_loss.item()
        if self.debug:
            print("[debug] BC Loss: {}".format(bc_loss.data))

        if self.add_rep_loss and (rep_dict['gt_rep'] is not None):
            # add an L2 loss between the data['task_emb'] and gt_rep
            criterion = nn.MSELoss()
            task_emb = rep_dict['task_emb']
            gt_rep = rep_dict['gt_rep']
            loss_rep = self.rep_loss_coef*criterion(task_emb, gt_rep.repeat(1, task_emb.shape[1], 1))
            loss += loss_rep

            self.log_info['rep_loss'] = loss_rep.item()
            self.log_info['rep_loss_orig'] = loss_rep.item()/self.rep_loss_coef
            if self.debug:
                print("[debug] Rep Loss: {}".format(loss_rep.data))
                print("[debug] Rep Loss Orig: {}".format(loss_rep.data/self.rep_loss_coef))

        if add_mfm:
            loss_mfm = self.mfm_head.loss_fn(
                        data=data,
                        feat=x[:,query_meta['vid_start_ind']:query_meta['vid_end_ind'],:],
            )
            self.log_info['mfm_loss'] = loss_mfm.item()
            if self.debug:
                print("MFM Loss", loss_mfm.data)
            loss += loss_mfm
        if add_mrm:
            loss_mrm = self.mrm_head.loss_fn(
                        data=data,
                        feat=x[:,query_meta['img_start_ind']:query_meta['img_end_ind'],:],
            )
            self.log_info['mrm_loss'] = loss_mrm.item()
            if self.debug:
                print("MRM Loss", loss_mrm.data)
            loss += loss_mrm
        if add_mim:
            loss_mim, mim_logits = self.mim_head.loss_fn(
                        data=data,
                        feat=x[:,query_meta['inst_start_ind']:query_meta['inst_end_ind'],:],
            )
            self.log_info['mim_loss'] = loss_mim.item()
            if self.debug:
                print("[debug] MIM loss: {}, BC Loss: {}".format(loss_mim.data, loss.data))
                mim_logits = mim_logits[:,0,:] ## just recreate the first masked token in the batch size
                _, pred_tokens = torch.max(F.softmax(mim_logits, dim=-1), dim=-1)
                pred_tokens = [token.cpu().int() for token in pred_tokens]
                pred_words = self.tz.decode(pred_tokens)
                print(data['gt_inst_ids'][:,0])
                print(pred_words)
            loss += loss_mim
        if add_mgm:
            loss_mgm, mgm_logits = self.mgm_head.loss_fn(
                        data=data,
                        feat=x[:,query_meta['gl_start_ind']:query_meta['gl_end_ind'],:],
            )
            self.log_info['mgm_loss'] = loss_mgm.item()
            if self.debug:
                print("[debug] MGM loss: {}, BC Loss: {}".format(loss_mgm.data, loss.data))
                mgm_logits = mgm_logits[:,0,:] ## just recreate the first masked token in the batch size
                _, pred_tokens = torch.max(F.softmax(mgm_logits, dim=-1), dim=-1)
                pred_tokens = [token.cpu().int() for token in pred_tokens]
                pred_words = self.tz.decode(pred_tokens)
                print(data['gt_gl_ids'][:,0])
                print(pred_words)
            loss += loss_mgm
        if add_maim:
            loss_maim = self.maim_head.loss_fn(
                        data=data,
                        feat=x[:,query_meta['ai_start_ind']:query_meta['ai_end_ind'],:],
            )
            self.log_info['maim_loss'] = loss_maim.item()
            if self.debug:
                print("MAIM Loss", loss_maim.data)
            loss += loss_maim
        if add_magm:
            loss_magm = self.magm_head.loss_fn(
                        data=data,
                        feat=x[:,query_meta['ag_start_ind']:query_meta['ag_end_ind'],:],
            )
            self.log_info['magm_loss'] = loss_magm.item()
            if self.debug:
                print("MAGM Loss", loss_magm.data)
            loss += loss_magm

        self.log_info['total_loss'] = loss.item()

        return loss

    def get_action(self, data):
        self.eval()
        with torch.no_grad():
            data = self.process_input(data, train_mode=False)
            x = self.spatial_encode(data)
            self.latent_queue.append(x)
            if len(self.latent_queue) > self.max_seq_len:
                self.latent_queue.pop(0)
            x = torch.cat(self.latent_queue, dim=1) # (B, T, H_all)
            x = self.temporal_encode(x, context_tokens=data['task_emb'])
            ### the decoder
            q_x, cross_attn_mask, self_attn_mask, query_meta = self.decoder.get_query_vec(
                        num_input_tokens=x.shape[1],
                        batch_size=x.shape[0],
                        lang_q_ind=[],
                        action_q_ind=x.shape[0]*[[i for i in range(x.shape[1])]],
            )
            x = self.decoder(inputs=x, query_vecs=q_x, cross_attn_mask=cross_attn_mask, self_attn_mask=self_attn_mask)
            ###
            dist = self.policy_head(x[:,-1])
        action = dist.sample().detach().cpu()
        return action.view(action.shape[0], -1).numpy()

    def reset(self):
        self.latent_queue = []
