CUDA_VISIBLE_DEVICES=0 python3 mutex/main_masked_modeling.py \
        benchmark_name=LIBERO_100 \
        policy.task_spec_modalities=gl_inst_img_vid_ai_ag \
        policy.add_mim=True policy.add_mgm=True policy.add_mrm=True \
        policy.add_mfm=True policy.add_maim=True policy.add_magm=True \
        folder=dataset-path \
        hydra.run.dir=experiments/mutex
