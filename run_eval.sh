MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0 python mutex/eval.py \
        benchmark_name=LIBERO_100 \
        folder=dataset-path \
        eval_spec_modalities=gl \
        experiment_dir=experiments/mutex \
        model_name=mutex_weights.pth
