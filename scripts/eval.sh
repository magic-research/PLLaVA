# export CUDA_VISIBLE_DEVICES=2,6,7
export OPENAI_API_KEY=...
num_frames=16
test_ratio=1

# 13b, uses offload thus saving the full model
model_dir=MODELS/plava-13b
weight_dir=MODELS/plava-13b
SAVE_DIR=test_results/test_plava_13b
lora_alpha=4
conv_mode=eval_vcgbench
python -m tasks.eval.vcgbench.plava_eval_vcgbench \
    --pretrained_model_name_or_path ${model_dir} \
    --save_path ${SAVE_DIR}/vcgbench \
    --num_frames ${num_frames} \
    --use_lora \
    --lora_alpha ${lora_alpha} \
    --weight_dir ${weight_dir} \
    --pooling_shape 16-12-12 \
    --test_ratio ${test_ratio} \
    --conv_mode ${conv_mode}

conv_mode=eval_mvbench
python -m tasks.eval.mvbench.plava_eval_mvbench \
    --pretrained_model_name_or_path ${model_dir} \
    --save_path ${SAVE_DIR}/mvbench \
    --use_lora \
    --lora_alpha ${lora_alpha} \
    --num_frames ${num_frames} \
    --weight_dir ${weight_dir} \
    --pooling_shape 16-12-12 \
    --conv_mode ${conv_mode}

onv_mode=eval_videoqabench
python -m tasks.eval.videoqabench.plava_eval_videoqabench \
    --pretrained_model_name_or_path ${model_dir} \
    --save_path ${SAVE_DIR}/videoqabench \
    --num_frames ${num_frames} \
    --use_lora \
    --lora_alpha ${lora_alpha} \
    --weight_dir ${weight_dir} \
    --test_ratio ${test_ratio} \
    --conv_mode ${conv_mode}


conv_mode=eval_recaption
python -m tasks.eval.recaption.plava_recaption \
    --pretrained_model_name_or_path ${model_dir} \
    --save_path ${SAVE_DIR}/recaption \
    --num_frames ${num_frames} \
    --use_lora \
    --weight_dir ${weight_dir} \
    --lora_alpha ${lora_alpha} \
    --test_ratio ${test_ratio} \
    --conv_mode ${conv_mode}


model_dir=MODELS/plava-7b
weight_dir=MODELS/plava-7b
SAVE_DIR=test_results/test_plava_7b
lora_alpha=4

conv_mode=eval_vcgbench
python -m tasks.eval.vcgbench.plava_eval_vcgbench \
    --pretrained_model_name_or_path ${model_dir} \
    --save_path ${SAVE_DIR}/vcgbench \
    --num_frames ${num_frames} \
    --use_lora \
    --lora_alpha ${lora_alpha} \
    --weight_dir ${weight_dir} \
    --pooling_shape 16-12-12 \
    --test_ratio ${test_ratio}


conv_mode=eval_mvbench
python -m tasks.eval.mvbench.plava_eval_mvbench \
    --pretrained_model_name_or_path ${model_dir} \
    --save_path ${SAVE_DIR}/mvbench \
    --use_lora \
    --lora_alpha ${lora_alpha} \
    --num_frames ${num_frames} \
    --weight_dir ${weight_dir} \
    --pooling_shape 16-12-12 


onv_mode=eval_videoqabench
python -m tasks.eval.videoqabench.plava_eval_videoqabench \
    --pretrained_model_name_or_path ${model_dir} \
    --save_path ${SAVE_DIR}/videoqabench \
    --num_frames ${num_frames} \
    --use_lora \
    --lora_alpha ${lora_alpha} \
    --weight_dir ${weight_dir} \
    --test_ratio ${test_ratio}

conv_mode=eval_recaption
python -m tasks.eval.recaption.plava_recaption \
    --pretrained_model_name_or_path ${model_dir} \
    --save_path ${SAVE_DIR}/recaption \
    --num_frames ${num_frames} \
    --use_lora \
    --lora_alpha ${lora_alpha} \
    --weight_dir ${weight_dir} \
    --test_ratio ${test_ratio}