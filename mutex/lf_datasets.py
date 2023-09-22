import torch
import copy
import cv2
import h5py
import imageio
import random
import numpy as np
from PIL import Image
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.file_utils as FileUtils

from torch.utils.data import Dataset
from mutex.utils import sample_frames
from robomimic.utils.dataset import SequenceDataset


"""
Helper function from Robomimic to read hdf5 demonstrations into sequence dataset

ISSUE: robomimic's SequenceDataset has two properties: seq_len and frame_stack,
we should in principle use seq_len, but the paddings of the two are different.
So that's why we currently use frame_stack instead of seq_len.
"""
def get_dataset(dataset_path, obs_modality, initialize_obs_utils=True,
                seq_len=1, frame_stack=1, filter_key=None,
                hdf5_cache_mode="low_dim", n_demos=None, *args, **kwargs):

    if initialize_obs_utils:
        ObsUtils.initialize_obs_utils_with_obs_specs({"obs": obs_modality})

    all_obs_keys = []
    for modality_name, modality_list in obs_modality.items():
        all_obs_keys += modality_list
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
            dataset_path=dataset_path,
            all_obs_keys=all_obs_keys,
            verbose=False)

    seq_len = seq_len
    filter_key = filter_key
    dataset = SequenceDataset(
                hdf5_path=dataset_path,
                obs_keys=shape_meta["all_obs_keys"],
                dataset_keys=["actions"],
                load_next_obs=False,
                frame_stack=frame_stack,
                seq_length=seq_len,               # length-10 temporal sequences
                pad_frame_stack=True,
                pad_seq_length=True,              # pad last obs per trajectory to ensure all sequences are sampled
                get_pad_mask=False,
                goal_mode=None,
                hdf5_cache_mode=hdf5_cache_mode,  # cache dataset in memory to avoid repeated file i/o
                hdf5_use_swmr=False,
                hdf5_normalize_obs=None,
                filter_by_attribute=filter_key,   # can optionally provide a filter key here
                n_demos=n_demos,
            )
    return dataset, shape_meta


class SequenceVLDataset(Dataset):
    def __init__(self, sequence_dataset, task_emb):
        self.sequence_dataset = sequence_dataset
        self.task_emb = task_emb
        self.n_demos = self.sequence_dataset.n_demos
        self.total_num_sequences = self.sequence_dataset.total_num_sequences

    def __len__(self):
        return len(self.sequence_dataset)

    def __getitem__(self, idx):
        return_dict = self.sequence_dataset.__getitem__(idx)
        return_dict["task_emb"] = self.task_emb
        return return_dict

class MLMTaskDataset(SequenceVLDataset):
    def __init__(
                    self,
                    sequence_dataset,
                    task_embs,
                    task_tokens,
                    gl_tokens,
                    inst_tokens,
                    gl_emb,
                    inst_emb,
                    ai_task_spec,
                    ag_task_spec,
                    visual_spec,  ## [length of video, h, w, 3]
                    cfg,
        ):
        self.task_tokens = task_tokens
        self.gl_tokens = gl_tokens
        self.inst_tokens = inst_tokens
        self.gl_emb = gl_emb
        self.inst_emb = inst_emb
        self.ai_task_spec = ai_task_spec
        self.ag_task_spec = ag_task_spec
        self.visual_spec = visual_spec ## visual_task_spec and visual_task_spec_mask
        self.t = cfg.policy.num_task_frames
        self.mask_lang=cfg.policy.add_mlm
        self.mask_vid=cfg.policy.add_mfm
        self.mask_img=cfg.policy.add_mrm
        self.mask_gl=cfg.policy.add_mgm
        self.mask_inst=cfg.policy.add_mim
        self.mask_ag=cfg.policy.add_magm
        self.mask_ai=cfg.policy.add_maim
        self.task_spec_modalities = cfg.policy.task_spec_modalities

        if self.mask_inst and self.inst_tokens is not None:
            self.mim_sample_indices = []
            for task_spec_id, inst_token in enumerate(self.inst_tokens['input_ids']): # number of possible instructions
                inst_list = []
                for inst_ind, sent_tokens in enumerate(inst_token): # number of sentences per instruction
                    if (torch.sum(self.inst_tokens['attention_mask'][task_spec_id][inst_ind].long())):
                        desc_list = []
                        for desc_ind,token in enumerate(sent_tokens): # number of tokens per sentence
                            if not (token in self.inst_tokens['stopword_tokens']):
                                desc_list.append(desc_ind)
                        assert len(desc_list) > 0, "No tokens found in sentence {}, {}, {}".format(task_spec_id, inst_ind, sent_tokens)
                        inst_list.append(desc_list)
                assert len(inst_list) > 0, "No tokens found in instruction"
                self.mim_sample_indices.append(inst_list)
        if self.mask_gl and self.gl_tokens is not None:
            self.mgm_sample_indices = []
            for task_spec_id, gl_token in enumerate(self.gl_tokens['input_ids']): # number of possible goal language
                gl_list = []
                for mgm_index, token in enumerate(gl_token):
                    if self.gl_tokens['attention_mask'][task_spec_id][mgm_index] and not (token in self.gl_tokens['stopword_tokens']):
                        gl_list.append(mgm_index)
                assert len(gl_list) > 0, "No tokens found in goal language."
                self.mgm_sample_indices.append(gl_list)

        self.cfg = cfg
        super().__init__(sequence_dataset, task_embs)

    def __len__(self):
        return len(self.sequence_dataset)

    def generate_visual_specifications(self):
        visual_dict = {}
        ## add all the visual specifications
        if 'vid' in self.task_spec_modalities:
            task_spec_id = random.randint(0, len(self.visual_spec['vid_task_spec'])-1)  # sample inclusive
            vid_task_spec = self.visual_spec['vid_task_spec'][task_spec_id].clone()
            vid_task_spec_mask = self.visual_spec['vid_task_spec_mask'][task_spec_id].clone()

            frame_idx = sample_frames(
                            num_frames=min(self.t-1, vid_task_spec.shape[0]-1),
                            vlen=vid_task_spec.shape[0]-1,
                            sample='rand' ## TODO: Make this rand
            ) ## should be 'rand' or 'uniform'
            frame_idx.append(vid_task_spec.shape[0]-1) ## add the last frame
            vid_spec = vid_task_spec[frame_idx]
            vid_spec_mask = vid_task_spec_mask[frame_idx]
            gt_vid_spec = vid_spec.clone() ## predicitng only average frame features

            if self.mask_vid:
                mfm_indices = [i for i, x in enumerate(vid_spec_mask) if x]
                mfm_indices = sorted(random.sample(mfm_indices, 1))
                vid_spec_mask[mfm_indices] = 0
            else:
                mfm_indices = []
            visual_dict['vid_spec'] = vid_spec
            visual_dict['vid_spec_mask'] = vid_spec_mask#.unsqueeze(dim=-1)
            visual_dict['mfm_indices'] = torch.Tensor(mfm_indices).long()
            visual_dict['gt_vid_spec'] = gt_vid_spec[mfm_indices]

        if 'img' in self.task_spec_modalities:
            task_spec_id = random.randint(0, len(self.visual_spec['img_task_spec'])-1)  # sample inclusive
            # don't clone here, since it is a very big tensor
            img_spec = self.visual_spec['img_task_spec'][task_spec_id] #.clone() ## [1, 50, 768]: Specifies the final goal
            #img_spec_mask = self.visual_spec['img_task_spec_mask'][task_spec_id].clone() # [1,50]
            # create using  torch ones to avoid in-place operations
            img_spec_mask = torch.ones(self.visual_spec['img_task_spec_mask'][task_spec_id].shape) ## [1,50]
            img_spec_mask = img_spec_mask.unsqueeze(dim=2).repeat(1,1,img_spec_mask.shape[-1]) ## [1,50,50], all for clip
            # don't clone here, since it is a very big tensor
            gt_img_spec = self.visual_spec['img_task_spec'][task_spec_id] #.clone()

            if self.mask_img:
                mrm_indices = range(1,img_spec_mask.shape[-1], 10) ## we cannot mask 0th feature because clip uses that as pooled feature, mask every 10th feature
                mrm_indices = sorted(random.sample(mrm_indices, 1))
                img_spec_mask[:, :, mrm_indices] = float("-inf") ## the way attention is defined in CLIP; attn_weights + attention_mask
            else:
                mrm_indices = []

            visual_dict['img_spec'] = img_spec
            visual_dict['img_spec_mask'] = img_spec_mask
            visual_dict['mrm_indices'] = torch.Tensor(mrm_indices).long()
            visual_dict['gt_img_spec'] = gt_img_spec[:, mrm_indices, :] ## [1, 1, 768]
        return visual_dict

    def generate_language_specifications(self):
        return_dict = {}
        if 'inst' in self.task_spec_modalities:
            task_spec_id = random.randint(0, len(self.inst_tokens['input_ids'])-1)  # sample inclusive
            input_ids = self.inst_tokens['input_ids'][task_spec_id].clone()
            gt_ids = self.inst_tokens['input_ids'][task_spec_id].clone()
            attention_mask = self.inst_tokens['attention_mask'][task_spec_id].clone()  ## don't make changes in the original task tokens
            input_mask = torch.sum(attention_mask, dim=-1) > 0
            total_valid_inst = torch.sum(input_mask)

            if self.mask_inst:
                ## mim_indices: gives indexes of the instructions
                ## desc_indices: gives indexes of the words positions corresponding to mim_indices
                mim_indices = sorted(random.sample(range(total_valid_inst), 2)) # pick any two instructions
                desc_indices = []
                for instruct_index in mim_indices:
                    ind = sorted(random.sample(self.mim_sample_indices[task_spec_id][instruct_index], 1))[0]
                    desc_indices.append(ind)
            else:
                return_dict["inst_emb"] = self.inst_emb[task_spec_id]
                mim_indices = []
                desc_indices = []

            attention_mask[mim_indices, desc_indices] = 0.

            inst_tokens = {}
            inst_tokens['attention_mask'] = attention_mask
            inst_tokens['input_ids'] = input_ids.long()
            return_dict['inst_tokens'] = inst_tokens
            return_dict['mim_indices'] = torch.Tensor(mim_indices).long()
            return_dict['desc_indices'] = torch.Tensor(desc_indices).long()
            return_dict['gt_inst_ids'] = gt_ids[mim_indices, desc_indices].long()
            return_dict['inst_emb_mask'] = torch.ones(return_dict['inst_tokens']['attention_mask'].shape[:-1])

        if 'gl' in self.task_spec_modalities:
            task_spec_id = random.randint(0, len(self.gl_emb)-1)  # sample inclusive
            if self.gl_tokens is None:
                return_dict["gl_emb"] = self.gl_emb[task_spec_id] ## adding time dimension
                # add time dimension if not present
                if len(return_dict["gl_emb"].shape) == 1:
                    return_dict['gl_emb'] = return_dict['gl_emb'].unsqueeze(dim=0)
                return return_dict

            input_ids = self.gl_tokens['input_ids'][task_spec_id].clone().long()
            gt_ids = self.gl_tokens['input_ids'][task_spec_id].clone()
            attention_mask = self.gl_tokens['attention_mask'][task_spec_id].clone()  ## don't make changes in the original task tokens

            if self.mask_gl:
                mgm_indices = sorted(random.sample(self.mgm_sample_indices[task_spec_id], 1)) ## TODO: Samples only one index at a time
            else:
                return_dict["gl_emb"] = self.gl_emb[task_spec_id].unsqueeze(dim=0) ## adding time dimension
                mgm_indices = []

            attention_mask[mgm_indices] = 0 ## change attention mask for tokens that are masked

            gl_tokens = {}
            gl_tokens['attention_mask'] = attention_mask
            gl_tokens['input_ids'] = input_ids ## not set to 0
            return_dict['gl_tokens'] = gl_tokens
            return_dict['mgm_indices'] = torch.Tensor(mgm_indices).long()
             ## Setting the order same as mgm_indices
            return_dict['gt_gl_ids'] = torch.Tensor([gt_ids[pred_val] for pred_val in  mgm_indices]).long() ## for the mgm loss
        return return_dict

    def generate_audio_specification(self):
        audio_dict = {}
        if 'ai' in self.task_spec_modalities:
            task_spec_id = random.randint(0, len(self.ai_task_spec['ai_task_spec'])-1)  # sample inclusive
            ai_spec = self.ai_task_spec['ai_task_spec'][task_spec_id].clone()
            ai_spec_mask = self.ai_task_spec['ai_task_spec_mask'][task_spec_id].clone()
            gt_ai_spec = self.ai_task_spec['ai_task_spec'][task_spec_id].clone()

            if self.mask_ai:
                maim_indices = [i for i,x in enumerate(ai_spec_mask) if x]
                maim_indices = sorted(random.sample(maim_indices, 1))
                ai_spec_mask[maim_indices] = 0
            else:
                maim_indices = []

            audio_dict['ai_task_spec'] = ai_spec
            audio_dict['ai_task_spec_mask'] = ai_spec_mask
            audio_dict['maim_indices'] = torch.Tensor(maim_indices).long()
            audio_dict['gt_ai_spec'] = gt_ai_spec[maim_indices]
        if 'ag' in self.task_spec_modalities:
            task_spec_id = random.randint(0, len(self.ag_task_spec['ag_task_spec'])-1)  # sample inclusive
            ag_spec = self.ag_task_spec['ag_task_spec'][task_spec_id].clone()
            ag_spec_mask = self.ag_task_spec['ag_task_spec_mask'][task_spec_id].clone()
            gt_ag_spec = self.ag_task_spec['ag_task_spec'][task_spec_id].clone()

            if self.mask_ag:
                magm_indices = [i for i,x in enumerate(ag_spec_mask) if x]
                magm_indices = sorted(random.sample(magm_indices, 1))
                ag_spec_mask[magm_indices] = 0
            else:
                magm_indices = []

            audio_dict['ag_task_spec'] = ag_spec
            audio_dict['ag_task_spec_mask'] = ag_spec_mask
            audio_dict['magm_indices'] = torch.Tensor(magm_indices).long()
            audio_dict['gt_ag_spec'] = gt_ag_spec[magm_indices]
        return audio_dict

    def __getitem__(self, idx):
        #return_dict = super().__getitem__(idx)
        return_dict = self.sequence_dataset.__getitem__(idx)

        return_dict.update(self.generate_language_specifications())

        visual_dict = self.generate_visual_specifications()
        return_dict.update(visual_dict)

        audio_dict = self.generate_audio_specification()
        return_dict.update(audio_dict)

        if 'lang' in self.task_spec_modalities:
            return_dict["task_emb"] = self.task_emb
            input_ids = self.task_tokens['input_ids'].clone()
            gt_ids = self.task_tokens['input_ids'].clone()
            attention_mask = self.task_tokens['attention_mask'].clone()  ## don't make changes in the original task tokens

            if self.mask_lang:
                mlm_indices = [i for i, x in enumerate(attention_mask) if x]
                #mlm_indices = random.sample(mlm_indices, int(len(mlm_indices)*0.15))
                mlm_indices = random.sample(mlm_indices, 1) ## TODO: Samples only one index at a time
                del return_dict['task_emb'] ## Removing task embeddings so that it does not use it in the algo.
            else:
                mlm_indices = []

            mask_token_id = self.task_tokens['mask_token_id']

            attention_mask[mlm_indices] = 0
            task_tokens = {}
            task_tokens['attention_mask'] = attention_mask
            task_tokens['input_ids'] = input_ids
            task_tokens['mlm_indices'] = torch.Tensor(mlm_indices).long()
             ## Setting the order same as mlm_indices
            task_tokens['gt_ids'] = torch.Tensor([gt_ids[pred_val] for pred_val in  mlm_indices]).long()
            return_dict['task_tokens'] = task_tokens

        return return_dict

class GroupedTaskDataset(Dataset):
    def __init__(self, sequence_datasets, task_embs):
        self.sequence_datasets = sequence_datasets
        self.task_embs = task_embs
        self.group_size = len(sequence_datasets)
        self.n_demos = sum([x.n_demos for x in self.sequence_datasets])
        self.total_num_sequences = sum([
            x.total_num_sequences for x in self.sequence_datasets])
        self.lengths = [len(x) for x in self.sequence_datasets]
        self.task_group_size = len(self.sequence_datasets)

        # create a map that maps the current idx of dataloader to original task data idx
        # imagine we have task 1,2,3, with sizes 3,5,4, then the idx looks like
        # task-1  task-2  task-3
        #   0       1       2
        #   3       4       5
        #   6       7       8
        #           9       10
        #           11
        # by doing so, when we concat the dataset, every task will have equal number of demos
        self.map_dict = {}
        sizes = np.array(self.lengths)
        row = 0
        col = 0
        for i in range(sum(sizes)):
            while sizes[col] == 0:
                col = col + 1
                if col >= self.task_group_size:
                    col -= self.task_group_size
                    row += 1
            self.map_dict[i] = (row, col)
            sizes[col] -= 1
            col += 1
            if col >= self.task_group_size:
                col -= self.task_group_size
                row += 1
        self.n_total = sum(self.lengths)

    def __len__(self):
        return self.n_total

    def __get_original_task_idx(self, idx):
        return self.map_dict[idx]
        #original_task_idx = 0
        #while idx > self.cum_lengths[original_task_idx]:
        #    original_task_idx += 1
        #if original_task_idx > 0:
        #    original_idx_in_task = idx - self.cum_lengths[original_task_idx-1]
        #else:
        #    original_idx_in_task = idx
        #return original_task_idx, original_idx_in_task

    def __getitem__(self, idx):
        oi, oti = self.__get_original_task_idx(idx)
        return_dict = self.sequence_datasets[oti].__getitem__(oi)
        return_dict["task_emb"] = self.task_embs[oti]
        return return_dict

class TruncatedSequenceDataset(Dataset):
    def __init__(self, sequence_dataset, buffer_size):
        self.sequence_dataset = sequence_dataset
        self.buffer_size = buffer_size

    def __len__(self):
        return self.buffer_size

    def __getitem__(self, idx):
        return self.sequence_dataset.__getitem__(idx)
