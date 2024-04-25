from tasks.train.instruction_data import *

# ========================= data ==========================
# train_corpus = "videochat2_instruction"
train_corpus = "videochat2_instruction_full"

train_file = "${available_corpus[${train_corpus}]}"  # for lazy evaluation
test_file = dict()
test_types = []
num_workers = 8
save_steps=10000
ckpt_steps=1000
stop_key = None
deepspeed=False
# ========================= input ==========================
num_frames = 16
num_frames_test = 1
batch_size = 1
gradient_accumulation_steps=16
max_txt_l = 512
max_train_steps=None
pre_text = False
gradient_checkpointing=False
inputs = dict(
    image_res=336,
    video_input=dict(
        num_frames="${num_frames}",
        sample_type="rand",
        num_frames_test="${num_frames_test}",
        sample_type_test="middle",
        random_aug=False,
    ),
    max_txt_l=dict(image="${max_txt_l}", video="${max_txt_l}"),
    batch_size=dict(image="${batch_size}", video="${batch_size}"),
    batch_size_test=dict(image="${batch_size}", video="${batch_size}"),
)

# ========================= model ==========================
model = dict(
    repo_id="llava-hf/llava-v1.6-vicuna-7b-hf",
    pretrained_path=None,
    load_from_origin=False,
    origin_vision="",
    origin_llm="",
    vision_encoder=dict(
        name="vit_l14", # somehow need this to tell the dataset the mean std of pretrained model
    ),
    torch_dtype='bfloat16',
    freeze_projector=False,
    freeze_lm=True,
    freeze_vision_tower=True,
    lora_target_modules=["q_proj", "v_proj"], # for llama/mistral/gemma
    use_lora=True,
    lora_r=128,
    lora_alpha=32,
    lora_dropout=0.05,
    num_frames="${num_frames}",
    pooling_method='avg',
    use_pooling=True,
    frame_shape=(24,24),
    pooling_shape=(16,8,8),
)
preprocess = dict(
    system="",
    mm_alone=True,
    random_shuffle=True,
    add_second_msg=True,
    roles=['USER:', 'ASSISTANT:'],
    end_signal=(' ', '</s>'),
    begin_signal='',
    dataset_image_placeholder='<Image></Image>',
    dataset_video_placeholder='<Video></Video>',
    image_token_index=32000,
    max_txt_l = "${max_txt_l}",
    ignore_index=-100, # same as torch softmax ignore index 
    center_pad=False,
    longest_edge=762,
    shortest_edge=336,
    clip_transform=False,
    num_frames="${num_frames}",
)


optimizer = dict(
    opt="adamW",
    lr=2e-5,
    opt_betas=[0.9, 0.999],  # default
    weight_decay=0.02,
    max_grad_norm=-1,  # requires a positive float, use -1 to disable
    # use a different lr for some modules, e.g., larger lr for new modules
    different_lr=dict(enable=False, module_names=[], lr=1e-3),
)

# scheduler = dict(sched="cosine", epochs=3, min_lr_multi=0.25, warmup_epochs=0.6)
# scheduler = dict(sched="cosine", epochs=3, min_lr_multi=0.25, warmup_epochs=0.6)
scheduler = dict(
    is_videochat2_custom=False,
    sched="cosine", 
    epochs=2, 
    warmup_ratio=0.2,
    min_lr_multi=0.25)

evaluate = False
deep_fusion = False
evaluation = dict(
    eval_frame_ensemble="concat",  # [concat, max, mean, lse]
    eval_x_only=False,
    k_test=128,
    eval_offload=True,  # offload gpu tensors to cpu to save memory.
)

fp16 = True
gradient_checkpointing = True

# ========================= wandb ==========================
wandb = dict(
    enable=False,
    entity="user",  # username or team name to store the runs, see https://docs.wandb.ai/ref/python/init
    project="videochat2",  # setup in your command line
)
dist_url = "env://"
device = "cuda"
mode = "it"

# ========================= others ==========================
output_dir = None  # output dir
resume = False  # if True, load optimizer and scheduler states as well
debug = False
log_freq = 5
metric_window_size=10 # window size for metric
seed = 42
report_to='tensorboard'
save_latest = True
auto_resume = True
pretrained_path = ""  # path to pretrained model weights, for resume only?
