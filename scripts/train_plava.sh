echo "PYTHONPATH: ${PYTHONPATH}"
which_python=$(which python)
echo "which python: ${which_python}"
export PYTHONPATH=${PYTHONPATH}:${which_python}
export PYTHONPATH=${PYTHONPATH}:.
echo "PYTHONPATH: ${PYTHONPATH}"

OUTPUT_DIR=./plava_video_outputs/test_train_7b_reconstruct

# # Naive Env
# rm -rf ${OUTPUT_DIR}
pooling_shape=(16,12,12)
accelerate launch --main_process_port 6876 --config_file scripts/accel_config_multigpu.yaml tasks/train/train_plava_nframe_accel.py \
    tasks/train/config_plava_nframe.py \
    output_dir ${OUTPUT_DIR} \
    train_corpus videochat2_video \
    save_steps 10000 \
    num_workers 8 \
    num_frames 16 \
    model.pooling_method avg \
    model.repo_id llava-hf/llava-v1.6-vicuna-7b-hf \
    model.use_lora True \
    model.use_cc False \
    model.pooling_shape $pooling_shape \
    optimizer.lr 2e-5 \
    scheduler.epochs 3 \
    scheduler.warmup_ratio 0.2 \
    scheduler.min_lr_multi 0.25 \
    scheduler.is_videochat2_custom True \
    preprocess.mm_alone False \
    preprocess.random_shuffle False \
    preprocess.add_second_msg False \
    train_corpus videochat2_instruction_debug

    