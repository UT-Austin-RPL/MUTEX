import os
# TODO: find a better way for this?
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import hydra
import json
import numpy as np
import pprint
import time
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
from mutex.embed_utils import get_visual_specifications_all, get_task_embs, get_audio_specification

def create_experiment_dir(cfg):
    if cfg.ckpt_dir is None:
        experiment_dir = os.getcwd()

        make_dir(experiment_dir)

        # look for the most recent run
        experiment_id = 0
        for path in Path(experiment_dir).glob('run_*'):
            if not path.is_dir():
                continue
            try:
                folder_id = int(str(path).split('run_')[-1])
                if folder_id > experiment_id:
                    experiment_id = folder_id
            except BaseException:
                pass
        experiment_id += 1

        experiment_dir += f"/run_{experiment_id:03d}"
        cfg.experiment_dir = experiment_dir
        cfg.experiment_name = "_".join(cfg.experiment_dir.split("/")[2:])
        os.makedirs(cfg.experiment_dir, exist_ok=True)
    else:
        ckpt_dir = cfg.ckpt_dir
        train_epochs = cfg.train.n_epochs
        with open(os.path.join(ckpt_dir, "config.json"), "r") as f:
            cfg = json.load(f)  # Overriding
            cfg = EasyDict(cfg)
        cfg.ckpt_dir = ckpt_dir  # overriding config file
        cfg.train.n_epochs = train_epochs
    return cfg

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(hydra_cfg):
    # preprocessing
    yaml_config = OmegaConf.to_yaml(hydra_cfg)
    cfg = EasyDict(yaml.safe_load(yaml_config))
    cfg = set_params(cfg)

    # print configs to terminal
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(cfg)

    # control seed
    control_seed(cfg.seed)

    # prepare lifelong learning
    cfg.folder = to_absolute_path(cfg.folder)
    cfg.bddl_folder = to_absolute_path(cfg.bddl_folder)
    cfg.init_states_folder = to_absolute_path(cfg.init_states_folder)

    benchmark_dict = bm.get_benchmark_dict()
    benchmark = benchmark_dict[cfg.benchmark_name.lower()]()
    n_manip_tasks = benchmark.n_tasks

    # prepare datasets from the benchmark
    task_list, task_demo_path_list = [], []
    manip_datasets = []
    descriptions = []
    dataset_inst, dataset_gl  = [], []

    inst_tokens, gl_tokens, task_tokens = None, None, None
    ag_task_specs, ai_task_specs = [], []
    task_visual_specifications = []
    inst_embs, gl_embs, task_embs = [], [], []

    t0 = time.time()
    for i in range(n_manip_tasks):
        task_spec_modalities = cfg.policy.task_spec_modalities.split('_')
        # currently we assume tasks from same benchmark have the same shape_meta
        task_i_dataset, shape_meta = get_dataset(
                n_demos=int(cfg.data.demos_per_task),
                dataset_path=os.path.join(cfg.folder,
                                          benchmark.get_task_demonstration(i)),
                obs_modality=cfg.data.obs.modality,
                initialize_obs_utils=(i==0),
                seq_len=cfg.data.seq_len,
                frame_stack=cfg.data.frame_stack
        )

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

    mode = 'train'
    if 'img' in task_spec_modalities or 'vid' in task_spec_modalities:
        print('Calculating visual embeddings')
        task_visual_specifications = get_visual_specifications_all(
                                cfg=cfg,
                                task_list=task_list,
                                benchmark_name=benchmark.name,
                                task_demo_path_list=task_demo_path_list,
        )
        print('Done')
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

    t1 = time.time()
    print(f'[info] load data time & task_embeddings (min) {(t1-t0)/60:.1f}')

    gsz = cfg.data.task_group_size
    assert gsz == 1, "We only use multitask setting."
    datasets = [MLMTaskDataset(
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
                    cfg=cfg,
            ) for i in range(len(manip_datasets))]
    n_demos = [data.n_demos for data in datasets]
    n_sequences = [data.total_num_sequences for data in datasets]

    n_tasks = n_manip_tasks // gsz # number of lifelong learning tasks
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

    # prepare experiment
    cfg = create_experiment_dir(cfg)

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
    cfg.device = 'cuda:0'  # If CUDA_VISIBLE_DEVICES are set correctly.
    algo = safe_device(eval(cfg.lifelong.algo)(n_tasks, cfg, logger=wandb), cfg.device)

    start_epoch = 0
    print(f"[info] start multitask learning with algo {cfg.lifelong.algo}")

    # save the experiment config file, so we can resume or replay later
    if cfg.ckpt_dir is None:
        with open(os.path.join(cfg.experiment_dir, "config.json"), "w") as f:
            json.dump(cfg, f, cls=NpEncoder, indent=4)

    assert cfg.lifelong.algo == 'Multitask'
    algo.start_task(-1)
    if cfg.ckpt_dir is not None:
        print("Loading from last checkpoint")
        start_epoch = algo.load_checkpoint()
    print("================= Resuming from epoch {}".format(start_epoch))
    algo.train()
    algo.learn_all_tasks(datasets, start_epoch=start_epoch)

    print("[info] finished learning\n")
    if cfg.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
