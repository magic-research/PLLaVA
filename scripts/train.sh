echo "PYTHONPATH: ${PYTHONPATH}"
which_python=$(which python)
echo "which python: ${which_python}"
export PYTHONPATH=${PYTHONPATH}:${which_python}
export PYTHONPATH=${PYTHONPATH}:.
echo "PYTHONPATH: ${PYTHONPATH}"

# SHARED ARGUMENTS

num_save_samples=80000
num_gpus=8
full_batch_size=128 # total batch size for gradient update
batch_size=4 # this is the batch size for each single gpu
save_steps=$[$num_save_samples/($batch_size*$num_gpus)] # save step is base on the "seen data" by the model. As the loop was counted base on data iteration, save step here doesn't take gradient accumulation into consideration.
ckpt_steps=$[$save_steps/10]
gradient_accumulation_steps=$[$full_batch_size/($batch_size*$num_gpus)] # some how the gradient accumulation steps should be computed as it was to pass in to Accelerate for not knowing the total GPUs, thus not able to compute afterwards. Align this with your accelerate configuration

# # Train 7B with LORA
# OUTPUT_DIR=./pllava_video_outputs/train_7b_lora
# repo_id=llava-hf/llava-v1.6-vicuna-7b-hf
# config_file=tasks/train/config_pllava_nframe.py
# accel_config=scripts/accel_config_multigpu.yaml # able to run with 80G A100
# extra_kwargs="\
#     model.repo_id $repo_id \
#     model.freeze_lm True \
#     model.use_lora True \
#     model.pretrained_path $pretrained_path \
# "


# Train 7B full model (vision & projector & language)
OUTPUT_DIR=./pllava_video_outputs/train_7b_trainall
repo_id=llava-hf/llava-v1.6-vicuna-7b-hf
config_file=tasks/train/config_pllava_nframe.py
accel_config=scripts/accel_config_deepspeed_zero3_trainLLM7B.yaml
extra_kwargs="\
    model.repo_id $repo_id \
    model.freeze_lm False \
    model.freeze_vision_tower False \
    model.use_lora False \
"

# # Train 34B with  LORA
# OUTPUT_DIR=./pllava_video_outputs/train_34b_lora
# repo_id=llava-hf/llava-v1.6-34b-hf
# config_file=tasks/train/config_pllava_nframe_yiprompt.py
# accel_config=scripts/accel_config_deepspeed_zero3_trainLLM34B.yaml
# extra_kwargs="\
#     model.repo_id $repo_id \
#     model.freeze_lm True \
#     model.use_lora True \
# "

echo "*************************PRE EXECUTION*******************************"
echo full batch size: $full_batch_size
echo instance batch size: $batch_size
echo num gpus: $num_gpus
echo gradient accumulation steps: $gradient_accumulation_steps
echo config file: $config_file
echo accelerate config file: $accel_config
echo "*********************************************************************"

# defualt configuration are as in ${config_file}, checkout for congiuraion
accelerate launch --main_process_port 6877 --config_file ${accel_config} tasks/train/train_pllava_nframe_accel.py  \
    ${config_file} \
    output_dir ${OUTPUT_DIR} \
    train_corpus videochat2_instruction_full \
    save_steps $save_steps \
    ckpt_steps $ckpt_steps \
    gradient_accumulation_steps $gradient_accumulation_steps \
    batch_size $batch_size \
    scheduler.is_videochat2_custom True \
    ${extra_kwargs}
