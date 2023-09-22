import copy
import pickle
import os
import glob
import torch
import numpy as np
from PIL import Image
from hydra.utils import get_original_cwd, to_absolute_path

import mutex
from datasets import Audio, Dataset
from transformers import logging as hf_logging
from transformers import AutoModel, pipeline, AutoTokenizer, CLIPTextModelWithProjection, CLIPFeatureExtractor
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from mutex.models.task_specs import CLIPVisionSliced

def get_audio_specification(benchmark_name, task_list, cfg, mode='train'):
    train_max_ts = int(0.8*cfg.n_ts_per_task)
    task_id_range = range(train_max_ts) if mode == 'train' else range(train_max_ts, cfg.n_ts_per_task)
    # Initialize WhisperProcessor and WhisperForConditionalGeneration
    tokenizer_str = "openai_whisper-small"
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(cfg.device)
    model.eval()
    model.config.forced_decoder_ids = None
    ai_task_spec_list = []
    ag_task_spec_list = []

    # Create decoder input ids
    decoder_input_ids = torch.Tensor([[1, 1]])*model.config.decoder_start_token_id
    decoder_input_ids = decoder_input_ids.long().to(cfg.device)
    for spec_type in ['ag', 'ai']:
        saved_emb_path = os.path.join(cfg.folder, benchmark_name, 'task_spec', f'{spec_type}_{tokenizer_str}_ts_mode_{mode}_emb.pt')
        if (not cfg.recalculate_ts_embs) and (os.path.exists(saved_emb_path)):
            print(f"[INFO]: Loading {spec_type} embeddings for ts mode {mode} from {saved_emb_path}")
            with open(saved_emb_path, 'rb') as fi:
                spec = pickle.load(fi)
            if spec_type == 'ai':
                ai_task_spec_list = spec
            if spec_type == 'ag':
                ag_task_spec_list = spec
            continue
        print(f"[WARNING]: Recalculating {spec_type} embeddings for ts mode {mode}. This may create a mismatch with train and eval.")
        for task_name in task_list:
            if spec_type  == 'ag':
                num_chunks = 4
            if spec_type == 'ai':
                num_chunks = 4
            encoder_feats_final = torch.empty((len(task_id_range), num_chunks, 768))
            for store_ind, ind in enumerate(task_id_range):
                audio_path = os.path.join(cfg.folder, benchmark_name, 'speech', task_name + f'_{spec_type}{ind:03d}.mp3')
                # Load audio file
                audio_dataset = Dataset.from_dict(
                        {"audio": [audio_path]},
                ).cast_column("audio", Audio(sampling_rate=16000))  # Sampling rate = 16k
                sample = audio_dataset[0]['audio']
                # Preprocess audio file
                input_features = processor(
                                    sample["array"],
                                    return_tensors="pt",
                                    sampling_rate=sample["sampling_rate"],
                ).input_features
                input_features = input_features.to(cfg.device)
                # Pass audio file through model
                with torch.no_grad():
                    model_out = model(input_features, decoder_input_ids=decoder_input_ids)
                encoder_feats = model_out.encoder_last_hidden_state
                # Reshape the tensor to [bs, n/4, 4, 768]
                encoder_feats = encoder_feats.view(
                        encoder_feats.size(0), encoder_feats.size(1) // num_chunks, num_chunks, encoder_feats.size(-1)
                )
                # Take the mean along the second (chunk) dimension
                encoder_feats = torch.mean(encoder_feats, dim=1)
                encoder_feats_final[store_ind] = encoder_feats[0]  # Remove batch size

            if cfg.train.debug:
                # Generate text from audio file
                predicted_ids = model.generate(input_features)
                transcription = processor.batch_decode(predicted_ids, skip_special_tokens=False)
                print("Audio Transcription")
                print("Model output logits shape: ", model_out.logits.shape)
                print(transcription)

                # Print output
                print("Encoder feature shapes: ", encoder_feats_final.shape)
            spec_dict = {
                    f'{spec_type}_task_spec': encoder_feats_final.cpu().clone(),
                    f'{spec_type}_task_spec_mask': torch.ones(encoder_feats_final.shape[:-1])
            }
            if spec_type == 'ai':
                ai_task_spec_list.append(spec_dict)
            if spec_type == 'ag':
                ag_task_spec_list.append(spec_dict)
            del model_out, encoder_feats_final
        # save the embeddings
        if spec_type == 'ai':
            with open(saved_emb_path, 'wb') as fi:
                pickle.dump(ai_task_spec_list, fi)
        if spec_type == 'ag':
            with open(saved_emb_path, 'wb') as fi:
                pickle.dump(ag_task_spec_list, fi)
    return ag_task_spec_list, ai_task_spec_list

def get_visual_specifications_all(benchmark_name, task_list, task_demo_path_list, cfg, mode='train'):
    train_max_ts = int(0.8*cfg.n_ts_per_task)
    task_id_range = range(train_max_ts) if mode == 'train' else range(train_max_ts, cfg.n_ts_per_task)

    tokenizer_str = cfg.tokenizer
    batch_size = cfg.train.batch_size
    tokenizer_str = tokenizer_str.replace('/', '_')
    saved_emb_path = os.path.join(cfg.folder, task_demo_path_list[0][:-5].split('/')[0], 'task_spec', f'visual_{tokenizer_str}_emb.pt')
    return_dict = {}
    if cfg.recalculate_ts_embs or (not os.path.exists(saved_emb_path)):
        print("[WARNING]: Recalculating visual embeddings. This may create mismatch between train and eval.")
        if cfg.visual_embedding_format == "clip":
            visual_preprocessor = CLIPFeatureExtractor.from_pretrained(cfg.tokenizer)
            hf_logging.set_verbosity_error()
            visual_emb_model = CLIPVisionSliced.from_pretrained(
                                                    cfg.tokenizer,
                                                    cache_dir=to_absolute_path("./clip")
            )
            visual_emb_model.eval()
            visual_emb_model.create_precomputable_models(layer_ind=cfg.policy.slice_model_ind)
            visual_emb_model = visual_emb_model.to(cfg.device)
            hf_logging.set_verbosity_warning()
        else:
            raise NotImplementedError
        for task_name, task_demo_path in zip(task_list, task_demo_path_list):
            vid_dir = os.path.join(cfg.folder, task_demo_path[:-5].split('/')[0], 'task_spec', task_demo_path[:-5].split('/')[1])
            img_task_spec_final, img_task_spec_mask_final = [], []
            vid_task_spec_final, vid_task_spec_mask_final = [], []
            for task_spec_id in range(cfg.n_ts_per_task): # Iterate over the different task specification examples, save everything
                visual_specs_path = os.path.join(vid_dir, f'vid_{task_spec_id:03d}')
                image_list = []
                for filename in sorted(glob.glob(f'{visual_specs_path}/*.png')):
                    im=Image.open(filename)
                    image_list.append(im)
                if cfg.visual_embedding_format == "clip":
                    visual_task_spec = visual_preprocessor(image_list, return_tensors='pt', padding=True)['pixel_values']
                elif cfg.visual_embedding_format == "r3m":
                    image_list = [torch.tensor(np.array(im)).permute(2,0,1) for im in image_list]
                    visual_task_spec = torch.stack(image_list, dim=0).to(cfg.device)
                else:
                    raise NotImplementedError
                visual_task_spec = visual_task_spec.to(cfg.device)
                visual_task_spec_list = []
                vid_task_spec_feat_list = []
                for i in range(0, visual_task_spec.shape[0], batch_size):
                    far_ind = min(visual_task_spec.shape[0], i+batch_size)
                    input_batch = visual_task_spec[i:far_ind]
                    with torch.no_grad():
                        output_batch = visual_emb_model.pre_compute_feats(input_batch)[0] ## [bs, 50, 512] 0th index has hidden embeds
                        vid_feats = visual_emb_model.post_compute_feats(output_batch)['image_embeds'] ## [bs, 512]
                    visual_task_spec_list.append(output_batch) ## for images
                    vid_task_spec_feat_list.append(vid_feats.detach()) ## stores the features, for videos

                visual_task_spec = torch.cat(visual_task_spec_list, dim=0) ## [T, 50, 512]
                img_task_spec_feat = visual_task_spec[-1:,:,:].detach().cpu() ## [1, 50, 512]
                vid_task_spec_feat = torch.cat(vid_task_spec_feat_list, dim=0).detach().cpu()  ## [T, 512]
                img_task_spec_final.append(img_task_spec_feat)
                vid_task_spec_final.append(vid_task_spec_feat)
                img_task_spec_mask_final.append(torch.ones(img_task_spec_feat.shape[:-1]))
                vid_task_spec_mask_final.append(torch.ones(vid_task_spec_feat.shape[:-1]))
            return_dict[task_name] = { # Save using task_name to maintain consistency accrross systems
                    'vid_task_spec': vid_task_spec_final,
                    'vid_task_spec_mask': vid_task_spec_mask_final,
                    'img_task_spec': img_task_spec_final,
                    'img_task_spec_mask': img_task_spec_mask_final
            }
            # save the dictionary to file using pickle
            with open(saved_emb_path, 'wb') as fi:
                pickle.dump(return_dict, fi)
    else:
        print(f"Reading embeddings from path {saved_emb_path}")
        # load the saved dictionary using pickle
        with open(saved_emb_path, 'rb') as fi:
            return_dict = pickle.load(fi)
    task_visual_specifications = []
    for task_name in task_list:
        for k,v in return_dict[task_name].items():
            return_dict[task_name][k] = [v[i] for i in task_id_range]
        task_visual_specifications.append(return_dict[task_name])
    return task_visual_specifications

def get_task_embs(cfg, descriptions, spec_type, mode='train'):
    # read stop words from file in mutex.__path__[0], english_stopwords.txt
    stop_words = []
    with open(os.path.join(mutex.__path__[0], 'english_stopwords.txt'), 'r') as fi:
        for line in fi:
            stop_words.append(line.strip())

    task_id_range = range(cfg.n_ts_per_task)
    save_embeddings = True
    if mode == 'train' or mode == 'eval':
        train_max_ts = int(0.8*cfg.n_ts_per_task)
        task_id_range = range(train_max_ts) if mode == 'train' else range(train_max_ts, cfg.n_ts_per_task, 1)
    else:
        raise NotImplementedError
    benchmark_name = cfg.benchmark_name.lower()
    tokenizer_str = cfg.lang_tokenizer
    tokenizer_str = tokenizer_str.replace('/', '_')
    saved_emb_path = os.path.join(cfg.folder, benchmark_name, 'task_spec', f'{spec_type}_{tokenizer_str}_ts_mode_{mode}_emb.pt')

    hf_logging.set_verbosity_error()
    if cfg.recalculate_ts_embs or (not os.path.exists(saved_emb_path)):
        print(f"[WARNING]: Calculating {spec_type} embeddings for ts mode {mode}")
        if cfg.lang_embedding_format == "clip":
            tz = AutoTokenizer.from_pretrained(cfg.lang_tokenizer)
            model = CLIPTextModelWithProjection.from_pretrained(cfg.lang_tokenizer, cache_dir=to_absolute_path("./clip")).eval()
        elif cfg.lang_embedding_format == "t5":
            tz = T5Tokenizer.from_pretrained(cfg.lang_tokenizer)
            model = T5ForConditionalGeneration.from_pretrained(cfg.lang_tokenizer).get_encoder().eval()
        else:
            raise NotImplementedError
        model = model.to(cfg.device)
        stopword_tokens = tz(stop_words, add_special_tokens=True)['input_ids']
        # make it a single list of tokens
        stopword_tokens = [item for sublist in stopword_tokens for item in sublist]
        # remove duplicates
        stopword_tokens = list(set(stopword_tokens))
        if spec_type == 'inst':
            task_embs = torch.empty((len(descriptions), len(task_id_range), cfg.data.max_instructs, 768))
            tokens = {
                        'input_ids': torch.empty((len(descriptions), len(task_id_range), cfg.data.max_instructs, cfg.data.max_word_len)), \
                        'attention_mask': torch.empty((len(descriptions), len(task_id_range), cfg.data.max_instructs, cfg.data.max_word_len)),
            }
            # iterate over tasks
            for task_ind, descriptions_task in enumerate(descriptions):
                # iterate over different descriptions for a task.
                for store_ind, ind in enumerate(task_id_range):
                    description = descriptions_task[ind]
                    token = tz(
                        text=description,                   # the sentence to be encoded
                        add_special_tokens=True,            # Add [CLS] and [SEP]
                        max_length=cfg.data.max_word_len,   # maximum length of a sentence
                        padding="max_length",
                        return_attention_mask=True,         # Generate the attention mask
                        return_tensors='pt',                # ask the function to return PyTorch tensors
                    )
                    # move token dictionary to cfg.device
                    for k, v in token.items():
                        token[k] = v.to(cfg.device)
                    if token['attention_mask'].size(0) > cfg.data.max_instructs:
                        print("[ERROR] Number of instructions are more than maximum allowed for task:", description)
                        raise Exception
                    elif token['attention_mask'].size(0) < cfg.data.max_instructs:
                        pad_len = cfg.data.max_instructs - token['attention_mask'].size(0)
                        for k, v in token.items():
                            token[k] = torch.cat((v,torch.zeros((pad_len, v.size(-1))).to(cfg.device)), dim=0).long()
                    if cfg.lang_embedding_format == "clip":
                        task_emb = model(**token)['text_embeds'].detach()
                    elif cfg.lang_embedding_format == "t5":
                        task_emb = model(**token)['last_hidden_state'].detach()
                        # mean over embeddings along the dimension=1
                        task_emb = torch.mean(task_emb, dim=-2)
                    tokens['input_ids'][task_ind, store_ind] = token['input_ids']
                    tokens['attention_mask'][task_ind, store_ind] = token['attention_mask']
                    task_embs[task_ind, store_ind] = task_emb
        elif spec_type == 'gl':
            task_embs = torch.empty((len(descriptions), len(task_id_range), 768))
            tokens = {
                        'input_ids': torch.empty((len(descriptions), len(task_id_range), cfg.data.max_word_len)), \
                        'attention_mask': torch.empty((len(descriptions), len(task_id_range), cfg.data.max_word_len)),
            }
            # iterate over tasks
            for task_ind, descriptions_task in enumerate(descriptions):
                descriptions_task = [descriptions_task[ind] for ind in task_id_range]
                token = tz(
                    text=descriptions_task,                 # the sentence to be encoded
                    add_special_tokens=True,                # Add [CLS] and [SEP]
                    max_length = cfg.data.max_word_len,     # maximum length of a sentence
                    padding="max_length",
                    return_attention_mask = True,           # Generate the attention mask
                    return_tensors = 'pt',                  # ask the function to return PyTorch tensors
                )
                # move token dictionary to cfg.device
                for k, v in token.items():
                    token[k] = v.to(cfg.device)
                if cfg.lang_embedding_format == "clip":
                    task_emb = model(**token)['text_embeds'].detach()
                elif cfg.lang_embedding_format == "t5":
                    task_emb = model(**token)['last_hidden_state'].detach()
                    # mean over embeddings along the dimension=-2
                    task_emb = torch.mean(task_emb, dim=-2)
                tokens['input_ids'][task_ind] = token['input_ids']
                tokens['attention_mask'][task_ind] = token['attention_mask']
                task_embs[task_ind] = task_emb
        else:
            raise NotImplementedError
        tokens["mask_token_id"] = torch.zeros((task_embs.shape[0],)).long() ## converts it to int64, there's not such thing as masking in clip
        # repeat stopword tokens for each task
        tokens['stopword_tokens'] = torch.tensor(stopword_tokens).long().unsqueeze(0).repeat(task_embs.shape[0], 1)
        tokens = {k: v.cpu().detach() for k,v in tokens.items()}
        task_embs = task_embs.cpu().detach()
        if save_embeddings:
            # if the parent directory ot saved_emb_path does not exist, create it
            if not os.path.exists(os.path.dirname(saved_emb_path)):
                os.makedirs(os.path.dirname(saved_emb_path))
            # save the dictionary to file using pickle
            # if the parent directory of saved_emb_path does not exist, create it
            if not os.path.exists(os.path.dirname(saved_emb_path)):
                os.makedirs(os.path.dirname(saved_emb_path))
            with open(saved_emb_path, 'wb') as fi:
                pickle.dump((task_embs, tokens), fi)
    else:
        print(f"[INFO]: Loading {spec_type} embeddings for ts mode {mode} from {saved_emb_path}")
        with open(saved_emb_path, 'rb') as fi:
            task_embs, tokens = pickle.load(fi)

    return task_embs, tokens
