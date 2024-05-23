echo "PYTHONPATH: ${PYTHONPATH}"
which_python=$(which python)
echo "which python: ${which_python}"
export PYTHONPATH=${PYTHONPATH}:${which_python}
export PYTHONPATH=${PYTHONPATH}:.
echo "PYTHONPATH: ${PYTHONPATH}"

OUTPUT_DIR=./pllava_video_outputs/test_train_7b_reconstruct

pooling_shape=(16,12,12)
num_save_samples=80000
num_gpus=8
full_batch_size=16
batch_size=1
save_steps=$[$num_save_samples/($batch_size*$num_gpus)]
ckpt_steps=$[$save_steps/10]
gradient_accumulation_steps=$[$full_batch_size/($batch_size*$num_gpus)]
echo $batch_size
echo $gradient_accumulation_steps
# repo_id=llava-hf/llava-v1.6-vicuna-7b-hf
# repo_id=llava-hf/llava-v1.6-vicuna-13b-hf
repo_id=llava-hf/llava-v1.6-34b-hf
accel_config=scripts/accel_config_deepspeed_zero3_trainLLM34B.yaml
extra_kwargs="\
    model.repo_id $repo_id \
    model.freeze_lm False \
    model.use_lora False \
    preprocess.image_token_index 64002 \
"
accelerate launch --main_process_port 6876 --config_file ${accel_config} tasks/train/train_pllava_nframe_accel.py  \
    tasks/train/config_pllava_nframe.py \
    output_dir ${OUTPUT_DIR} \
    train_corpus videochat2_instruction_debug \
    save_steps $save_steps \
    ckpt_steps $ckpt_steps \
    num_workers 8 \
    num_frames 16 \
    gradient_accumulation_steps $gradient_accumulation_steps \
    batch_size $batch_size \
    model.pooling_method avg \
    model.use_pooling True \
    gradient_checkpointing True \
    preprocess.center_pad False \
    preprocess.clip_transform False \
    optimizer.lr 2e-5 \
    scheduler.epochs 3 \
    scheduler.warmup_ratio 0.2 \
    scheduler.min_lr_multi 0.25 \
    model.pooling_shape $pooling_shape \
    scheduler.is_videochat2_custom True \
    preprocess.mm_alone False \
    preprocess.random_shuffle False \
    preprocess.add_second_msg False \
    ${extra_kwargs}
