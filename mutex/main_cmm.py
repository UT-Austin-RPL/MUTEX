import os
# TODO: find a better way for this?
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import hydra
import json
import numpy as np
import pprint
import time
import torch
import wandb
import yaml
from easydict import EasyDict
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from pathlib import Path

from libero.libero import benchmark as bm
from mutex.algos import Multitask
from mutex.lf_datasets import get_dataset, MLMTaskDataset
from mutex.utils import control_seed, safe_device, torch_load_model, \
                           NpEncoder, make_dir, set_params
from mutex.embed_utils import get_task_embs, get_audio_specification
from mutex.embed_utils import get_visual_specifications_all, get_task_embs, get_audio_specification

def get_model_weights(_model_weights, algo, cfg):
    # remove all the weights that are not in the current model
    # print all weights not present in current model
    model_weights = {}
    if cfg.num_gpus > 1:
        # add module. to all the weights of model_weights
        for k,v in _model_weights.items():
            model_weights['module.'+k] = _model_weights[k]
    else:
        model_weights = _model_weights
    #print("---------------------------- Weights not present in current model:")
    for k in model_weights.keys():
        if k not in algo.policy.state_dict():
            if not ('module.' + k) in algo.policy.state_dict():
                print(k)
    #print("++++++++++++++++++++++++++++ Weights not found in model_weights dict:")
    for k in algo.policy.state_dict().keys():
        if k not in model_weights:
            if 'cross_modal' in k:
                # Initialize cross_modal projection layers to identity function
                # if k is weight, initialize it with identity matrix
                # if k is bias, initialize it with zeros
                if 'weight' in k:
                    model_weights[k] = torch.eye(algo.policy.state_dict()[k].shape[0]).to(cfg.device)
                elif 'bias' in k:
                    model_weights[k] = torch.zeros(algo.policy.state_dict()[k].shape[0]).to(cfg.device)
                else:
                    raise NotImplementedError
            else:
                print("[ERROR] {} not found in model_weights dict".format(k))
                raise ValueError

    model_weights = {k: v for k, v in model_weights.items() if k in algo.policy.state_dict()}
    return model_weights

def update_cmm_configs(cfg, cmm_cfg):
    # prepare multitask learning
    cfg.device = cmm_cfg.device
    cfg.folder = to_absolute_path(cmm_cfg.folder)
    cfg.bddl_folder = to_absolute_path(cmm_cfg.bddl_folder)
    cfg.init_states_folder = to_absolute_path(cmm_cfg.init_states_folder)
    train_benchmark_name = cfg.benchmark_name
    cfg.benchmark_name = cmm_cfg.benchmark_name if cmm_cfg.benchmark_name is not None else cfg.benchmark_name
    cfg.experiment_dir = cmm_cfg.experiment_dir
    cfg.num_gpus = cmm_cfg.num_gpus
    cfg.save_ckpt = False
    cfg.save_best_model = False
    cfg.recalculate_ts_embs = False
    cfg.wandb_project = cmm_cfg.wandb_project
    cfg.wandb_mode = cmm_cfg.wandb_mode
    cfg.use_wandb = cmm_cfg.use_wandb
    cfg.train.debug = cmm_cfg.debug
    cfg.experiment_name = f'cmm_{cfg.benchmark_name}_lr{cmm_cfg.lr}_' + cfg.experiment_name

    # train detail overriding
    cfg.policy.add_rep_loss = True # the whole point of finetuning is with rep loss
    cfg.policy.sg_gt_rep = cmm_cfg.sg_gt_rep
    cfg.policy.rep_loss_coef = cmm_cfg.rep_loss_coef
    cfg.train.n_epochs = cmm_cfg.n_epochs
    cfg.train.batch_size = cmm_cfg.batch_size
    cfg.train.val_every = cmm_cfg.val_every
    cfg.train.optimizer.kwargs.lr = cmm_cfg.lr
    cfg.train.scheduler.kwargs.eta_min = cmm_cfg.lr / 10.0
    cfg.train.num_workers = cmm_cfg.num_workers

    # turn off all the masked modeling losses
    cfg.policy.add_mim = False
    cfg.policy.add_mgm = False
    cfg.policy.add_mrm = False
    cfg.policy.add_mfm = False
    cfg.policy.add_maim = False
    cfg.policy.add_magm = False

    # Add cross modal projection layers
    cfg.policy.projection_layer.network_kwargs.inst_transform_kwargs.network_kwargs.add_cross_modal_layer = cmm_cfg.add_cross_modal_layers
    cfg.policy.projection_layer.network_kwargs.gl_transform_kwargs.network_kwargs.add_cross_modal_layer = cmm_cfg.add_cross_modal_layers
    cfg.policy.projection_layer.network_kwargs.ai_transform_kwargs.network_kwargs.add_cross_modal_layer = cmm_cfg.add_cross_modal_layers
    cfg.policy.projection_layer.network_kwargs.ag_transform_kwargs.network_kwargs.add_cross_modal_layer = cmm_cfg.add_cross_modal_layers
    cfg.policy.projection_layer.network_kwargs.img_transform_kwargs.network_kwargs.add_cross_modal_layer = cmm_cfg.add_cross_modal_layers

    return cfg

@hydra.main(config_path="../configs/cross_modal", config_name="cross_modal", version_base=None)
def main(hydra_cfg):
    with open(os.path.join(hydra_cfg.experiment_dir, "config.json"), "r") as f:
        cfg = json.load(f)

    # preprocessing
    yaml_config = OmegaConf.to_yaml(hydra_cfg)
    cmm_cfg = EasyDict(yaml.safe_load(yaml_config))
    cfg = EasyDict(cfg)

    # print configs to terminal
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(cfg)

    # control seed
    control_seed(cfg.seed)

    cfg = update_cmm_configs(cfg, cmm_cfg)

    epoch_num = int(cmm_cfg.cmm_type.split("_")[-1])
    model_path = os.path.join(cmm_cfg.experiment_dir, "models", f"multitask_model_ep{epoch_num:03d}.pth")
    assert os.path.exists(model_path), 'Model path {} does not exist!'.format(model_path)

    benchmark_dict = bm.get_benchmark_dict()
    benchmark = benchmark_dict[cfg.benchmark_name.lower()]()
    n_manip_tasks = benchmark.n_tasks

    # prepare datasets from the benchmark
    task_list, task_demo_path_list = [], []
    manip_datasets = []
    descriptions = []
    dataset_inst, dataset_gl  = [], []
    inst_tokens, gl_tokens, task_tokens = None, None, None
    inst_embs, gl_embs, task_embs = [], [], []
    ag_task_specs, ai_task_specs = [], []
    task_visual_specifications = []

    t0 = time.time()
    for i in range(n_manip_tasks):
        task_spec_modalities = cfg.policy.task_spec_modalities.split('_')
        # currently we assume tasks from same benchmark have the same shape_meta
        task_i_dataset, shape_meta = get_dataset(
                n_demos=int(cfg.data.demos_per_task),
                dataset_path=os.path.join(cfg.folder, benchmark.get_task_demonstration(i)),
                obs_modality=cfg.data.obs.modality,
                initialize_obs_utils=(i==0),
                seq_len=cfg.data.seq_len,
                frame_stack=cfg.data.frame_stack)

        # add language to the vision dataset, hence we call vl_dataset
        task_name = benchmark.get_task(i).name
        task_description = benchmark.get_task(i).language
        instructions = benchmark.get_task(i).instructions
        goal_language = benchmark.get_task(i).goal_language

        task_list.append(task_name)
        task_demo_path_list.append(benchmark.get_task_demonstration(i))
        descriptions.append(task_description)
        manip_datasets.append(task_i_dataset)
        dataset_inst.append(instructions)
        dataset_gl.append(goal_language)


    if 'img' in task_spec_modalities or 'vid' in task_spec_modalities:
        print('Calculating visual embeddings')
        task_visual_specifications = get_visual_specifications_all(
                                cfg=cfg,
                                task_list=task_list,
                                benchmark_name=benchmark.name,
                                task_demo_path_list=task_demo_path_list,
        )
        print('Done')

    mode = 'train'
    if ('ag' in task_spec_modalities) or ('ai' in task_spec_modalities):
        print('Calculating audio embeddings')
        ag_task_specs, ai_task_specs = get_audio_specification(
                                benchmark_name=benchmark.name,
                                task_list=task_list,
                                cfg=cfg,
                                mode=mode,
        )
        print('Done')
    if 'inst' in task_spec_modalities:
        print('Calculating instruction embeddings')
        inst_embs, inst_tokens = get_task_embs(cfg, dataset_inst, spec_type='inst', mode=mode)
        print('Done')
    if 'gl' in task_spec_modalities:
        print('Calculating goal language embeddings')
        gl_embs, gl_tokens = get_task_embs(cfg, dataset_gl, spec_type='gl', mode=mode)
        print('Done')
    if 'lang' in task_spec_modalities:
        print('Calculating task embeddings')
        task_embs, task_tokens = get_task_embs(cfg, descriptions, spec_type='lang', mode=mode)
        print('Done')

    t1 = time.time()
    print(f'[info] load data time & task_embeddings (min) {(t1-t0)/60:.1f}')

    gsz = cfg.data.task_group_size
    assert gsz == 1, "We only use multitask setting."
    datasets = [
            MLMTaskDataset(
                    sequence_dataset=manip_datasets[i],
                    task_embs=task_embs[i] if len(task_embs) > 0 else None,
                    task_tokens={k: v[i] for k, v in task_tokens.items()} if task_tokens is not None else None,
                    gl_tokens={k: v[i] for k, v in gl_tokens.items()} if gl_tokens is not None else None,
                    inst_tokens={k: v[i] for k, v in inst_tokens.items()} if inst_tokens is not None else None,
                    gl_emb=gl_embs[i] if len(gl_embs) > 0 else None,
                    inst_emb=inst_embs[i] if len(inst_embs) > 0 else None,
                    ai_task_spec=ai_task_specs[i] if len(ai_task_specs) > 0 else None,
                    ag_task_spec=ag_task_specs[i] if len(ag_task_specs) > 0 else None,
                    visual_spec=task_visual_specifications[i] if len(task_visual_specifications) > 0 else None,
                    cfg=cfg) for i in range(len(manip_datasets))
    ]
    n_demos = [data.n_demos for data in datasets]
    n_sequences = [data.total_num_sequences for data in datasets]

    n_tasks = n_manip_tasks // gsz # number of multitask learning tasks
    cfg.shape_meta = shape_meta
    print("\n=================== Lifelong Benchmark Information  ===================")
    print(f" Name: {benchmark.name}")
    print(f" # Tasks: {n_manip_tasks // gsz}")
    for i in range(n_tasks):
        print(f"    - Task {i+1}:")
        for j in range(gsz):
                print(f"        {benchmark.get_task(i*gsz+j).language}")
    print(" # demonstrations: " + " ".join(f"({x})" for x in n_demos))
    print(" # sequences: " + " ".join(f"({x})" for x in n_sequences))
    print("=======================================================================\n")

    if cfg.use_wandb:
        import wandb
        num_attempts = 10
        for attempt in range(num_attempts):
            try:
                # set up wandb
                wandb.init(project=cfg.wandb_project, dir=cfg.experiment_dir, resume=not (cfg.ckpt_dir is None), config=cfg, mode="offline" if attempt == num_attempts-1 else cfg.wandb_mode)
                wandb.run.name = cfg.experiment_name
                wandb.define_metric("epoch")
                wandb.define_metric("train/*", "epoch")
                wandb.define_metric("val/*", "epoch")
                break
            except:
                print("wandb initialization, attempt #{}".format(attempt + 1))
                wandb = None
                time.sleep(30)
    else:
        wandb = None

    # define multitask algorithm
    algo = safe_device(eval(cfg.lifelong.algo)(n_tasks, cfg, logger=wandb), cfg.device)
    start_epoch = 0

    print(f"Loading model from {model_path}")
    _model_weights = torch_load_model(model_path)[0]
    model_weights = get_model_weights(_model_weights, algo, cfg)
    algo.policy.load_state_dict(model_weights)

    print(f"[info] start lifelong learning with algo {cfg.lifelong.algo}")

    assert cfg.lifelong.algo == 'Multitask'
    algo.start_task(-1)
    print("================= Resuming from epoch {}".format(start_epoch))
    algo.train()

    # freeze gradients for all parameters that do not have 'projection_layers' in their name
    for name, param in algo.policy.named_parameters():
        if 'projection_layers' not in name:
            # only projection layers are trained during cross modal matching
            param.requires_grad = False
        else:
            # SharedMLP should not change during cross modal matching
            if name.startswith('projection_layers.linear_projection'):
                param.requires_grad = False

    if cfg.train.debug:
        # store all the parameter weights:
        param_weights = {}
        for name, param in algo.policy.named_parameters():
            param_weights[name] = param.data.clone().detach()

    model_prefix = f'cmm_{cfg.benchmark_name}_'
    algo.policy.cross_modal_matching = True
    algo.learn_all_tasks(datasets, start_epoch=start_epoch, model_prefix=model_prefix)

    if cfg.train.debug:
        # find parameters that have changed
        for name, param in algo.policy.named_parameters():
            # check if equal
            if not torch.equal(param.data, param_weights[name]):
                print(f"Parameter {name} has changed")

    print("[info] finished learning\n")
    if cfg.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
