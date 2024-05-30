export OPENAI_API_KEY=...
# Shared
num_frames=16
model_dir=llava-hf/llava-v1.6-vicuna-7b-hf

# # All Trained
# SAVE_DIR=test_results/${weight_dir}
# shared_args="\
#     --pretrained_model_name_or_path ${model_dir} \
#     --num_frames ${num_frames} \
#     --pooling_shape 16-12-12 \

# " # this will be appended to each ecaluation command

# for lora
weight_dir=pllava_video_outputs/test_train_7b_lora/pretrained_step0.1600M
lora_alpha=4
SAVE_DIR=test_results/${weight_dir}
shared_args="\
    --use_lora \
    --lora_alpha ${lora_alpha} \
    --weight_dir ${weight_dir} \
    --pretrained_model_name_or_path ${model_dir} \
    --num_frames ${num_frames} \

" # this will be appended to each ecaluation command


# VCG
python -m tasks.eval.vcgbench.pllava_eval_vcgbench \
    --conv_mode eval_vcg_llavanext \
    --save_path ${SAVE_DIR}/vcgbench \
    --pooling_shape 16-12-12 \
    ${shared_args}

# MVbench
python -m tasks.eval.mvbench.pllava_eval_mvbench \
    --conv_mode eval_mvbench_llavanext \
    --save_path ${SAVE_DIR}/mvbench \
    --pooling_shape 16-12-12 \
    ${shared_args}

# VideoQA
python -m tasks.eval.videoqabench.pllava_eval_videoqabench \
    --conv_mode eval_videoqa_llavanext \
    --save_path ${SAVE_DIR}/videoqabench \
    --test_ratio 2000 \
    --test_dataset MSVD_QA-MSRVTT_QA-ActivityNet-TGIF_QA \
    ${shared_args}

# # Recaption
# python -m tasks.eval.recaption.pllava_recaption \
#     --save_path ${SAVE_DIR}/recaption \
#     --conv_mode eval_recaption_llavanext \
#     ${shared_args}
