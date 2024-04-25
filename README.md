<div align="center">

<h2><a href="https://arxiv.org/abs/2311.17005">PLLaVA : Parameter-free LLaVA Extension from Images to Videos for Video Dense Captioning</a></h2>

[Lin Xu](https://scholar.google.com/citations?user=_Gu69coAAAAJ), [Yilin Zhao](https://ermu2001.github.io/me.io/), [Daquan Zhou](https://scholar.google.com/citations?user=DdCAbWwAAAAJ), [Zhijie Lin](https://scholar.google.com/citations?user=xXMj6_EAAAAJ), [See-Kiong Ng](https://scholar.google.com/citations?user=_wsommYAAAAJ), [Jiashi Feng](https://scholar.google.com.sg/citations?user=Q8iay0gAAAAJ&hl=en)
</div>

[![Paper](https://img.shields.io/badge/cs.CV-2311.17005-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2311.17005)
[![YouTube Video](https://img.shields.io/badge/YouTube-Video-red)]()
[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm-dark.svg)](https://huggingface.co/spaces)
[![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm-dark.svg)](https://huggingface.co/models)

![](assert/logo.png)
## Overview
Welcome to PLAVA!

The primary purpose of this repository is to support research and the development of prototype models. It is designed to facilitate ease of experimentation and enable a clear overview of results. Please note that this section is currently undergoing development and reconstruction.

It's important to mention that we have not optimized the response speed of the application or the frontend logic. Our goal is to maintain simplicity, clarity, and ease of development, making it accessible for both researchers and students. If you have suggestions or want to enhance the application's performance, please feel free to contact us or contribute to the project.


We've briefly introduce our work in [PLAVA](#%EF%B8%8F-plava). For more details, feel free to read our paper. Checkout [Usage](#hammer-usage) to start using this repo. If you felt our works interesting, please star us, your support is all we want. If you find our work helpful, feel free to [cite](#page_facing_up-citation) us directly.

## :fire: Updates
- **2024/4/24**: Release:
    - We are releasing our code/models/datasets.

## 🏖️ PLAVA
TODO: Introduction to PLAVA


## :hammer: Usage
This section provides guidance on how to run, train, and evaluate our models.


### Install
First you will need to set up the environment, and download some pretrained weights.

This repo is built up using [transformers](https://github.com/huggingface/transformers) for model construction along with [accelerate](https://github.com/huggingface/transformers) for distributed training. Follow the instruction to install the needed environment.

1. Firstly, install [pytorch](https://pytorch.org/) from the official website. The code runs on torch 2.2.*, cu118 or cu122.
```
torch                       2.2.1+cu118
torchaudio                  2.2.1+cu118
torchvision                 0.17.1+cu118
```

2. Install the other requirements, This file list serve for torch 2.2.1+cu118 with driver version 11.6
```
pip install -r requirements.no_torch.txt
```

3. Prepare the model.
We prefer to have huggingface models explicitly download to a MODELS directory. However, if you are familiar with huggingface-hub usage, feel free to organize the model yourself.
```
python python_scripts/hf.py
```
With the above steps, you should be able to proceed on the following usages.

### Run Application
Aside from this
To run our models, make sure you have downloaded a model pretrained weights from the huggingface spaces. Then, run the following scripts with the corresponding path input.
- model_dir: your model directory, one with config.json as compatible with transformers
- weights_dir: your weights directory. could be the same as model_dir, but if you have a weights directory for the lora weights, you should set this weights_dir to that directory to load the lora weights. Also, it would need to contain a config.json file under.
```
model_dir="your model directory"
weights_dir="your weights directory (Could be the same as model_dir, but if you have a weights directory for the lora weights, this weights_dir should point to that directory to load the lora weights)"
bash scripts/demo.sh ${model_dir} ${weights_dir}
```
Now checkout the application demo and try play with PLAVA!

### Train
Follow the following steps to reproduce our results or train your own variant:

#### 1. Data Preparation

To train our model from a starting Image-aligned Vision LLM, you would need to download the data first. Our data set up is mainly based on the original Videochat2's training data. Checkout [Instruction Data](./DATA.md) to prepare the instruction training data. Eventually, you should have a folder organized as following:
```TODO:
(magic_py310) (base) xulin@9f0c9285c368:~/yilin/magic_video$ ls -l DATAS/TRAIN_TEST
total 20
-rw-r--r--  1 xulin users  123 Jan 31 15:57 DATA.md
-rw-r--r--  1 xulin users 2688 Jan 31 15:56 how_to_use.py
drwxr-xr-x 11 xulin users 4096 Jan 31 15:50 images
drwxr-xr-x  4 xulin users 4096 Jan 31 15:53 magic_jsons
drwxr-xr-x 12 xulin users 4096 Jan 30 17:00 videos
```

#### 2. Start Training
Now you're only a few step away from starting the training. Follow the instructions:

##### Setup Accelerator
Customize a accelerate training config. For example, a simple config using multiple gpus with no distribution strategy (only torch DDP) would look like:
```yaml
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 8
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
``` 
Checkout out the [Accelerate](https://huggingface.co/docs/accelerate/index) documents for more details.

##### Overwatch the training configuration
Next, you should go over a basic training configuration of the training process in [here](tasks/train/config_magic_nframe.py). Then passing this file as the first arg to the training script would utilize every arguments in the file. You can customize some of the hyper parameters for your own training process by passing them in the format of "key" "value" pair in the following arguments. A example training scripts could be find TODO: [here](scripts/train_magic.sh). 

This part of configuration is mostly based on the original [Videochat2](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat2). Salute to those fantastic researchers & engineers.

With the above steps, you would be able to start the training process. The output would be well organized in the output directory, each a qualified model directory to pass in to demo as weights_dir, since we are only saveing the lora weights and projector weights to avoide redundancy.

### Evaluation
This section mainly introduce how to reproduce the evaluation or evaluate your own model.

#### Set up Evaluation Data
Make sure you set up the "DATAS" directory as, then you would be able to run the inference with fortune!
```
DATAS/:
      TGIF_FrameQA.csv
DATAS/VideoQA:
DATAS/VideoQA/TGIF_QA:
                     test_a.json
DATAS/VideoQA/TGIF_QA/videos:
                            tumblr_m4387mGrlc1r6m5e8o1_250.gif
DATAS/VideoQA/TGIF_QA/videos_mp4:
                                tumblr_m4387mGrlc1r6m5e8o1_250.mp4
DATAS/VideoQA/TGIF_QA/video_gif:
                               tumblr_m4387mGrlc1r6m5e8o1_250.gif
DATAS/VideoQA/MSVD_Zero_Shot_QA:
                               raw-captions.pkl
DATAS/VideoQA/MSVD_Zero_Shot_QA/videos:
                                      -4wsuPCjDBc_5_15.avi
DATAS/VideoQA/MSVD_Zero_Shot_QA/msvd_qa:
                                       test.jsonl
DATAS/VideoQA/ActivityNet:
                         test_a.json
DATAS/VideoQA/ActivityNet/all_test:
                                  v_--tFD65KaK4.mp4
DATAS/VideoQA/ActivityNet/all_test/v_Fvm9BuMz0yE.mp4:
DATAS/VideoQA/ActivityNet/all_test/v_Fvm9BuMz0yE.mp4/all_test:
                                                             v_10fX73-AXcg.mp4
DATAS/VideoQA/MSRVTT_Zero_Shot_QA:
                                 MSRVTT_JSFUSION_test.csv
DATAS/VideoQA/MSRVTT_Zero_Shot_QA/videos:
                                        test_list_new.txt
DATAS/VideoQA/MSRVTT_Zero_Shot_QA/videos/all:
                                            video0.mp4
DATAS/VideoQA/MSRVTT_Zero_Shot_QA/videos/all/images:
DATAS/VideoQA/MSRVTT_Zero_Shot_QA/msrvtt_qa:
                                           test.jsonl

DATAS/MVBench:
             ...

DATAS/Recaption/Inter4K:
                       annotations.json
DATAS/Recaption/Inter4K/60fps:
DATAS/Recaption/Inter4K/60fps/UHD:
                                 1.mp4

```

#### Start Evaluate
Once you have construted the evaluation data, you can start the evaluation as in [here](scripts/eval.sh)
```
bash scripts/eval.sh
```
Same as running the demo, you would need to determine the model_dir and weights_dir to evaluate the model. Feel free to comment out some commands and produce partial evaluation. TODO: refine eval_all.sh

#### Overwatch the Results
The evaluation results would be shown to you with our results gallery demo:
```
bash scripts/gallery.sh
```

Feel free to use the compare version to compare differnt models' results or use the single gallery version to checkout one model's results. They are basically the same. Checkout the [script](scripts/gallery.sh) for more details


# :page_facing_up: Citation

If you find this project useful in your research, please consider cite:
```BibTeX

```

# :dizzy: Acknowledgement
This code base is mainly built upon [Videochat2](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat2).

We would also like to recognize and commend the following open source projects, thank you for your great contribution to the open source community:

- [LLaVA](https://github.com/haotian-liu/LLaVA): Fantastic Open Source Vision LLM Model. 
- [VideoChatGPT](https://github.com/mbzuai-oryx/Video-ChatGPT/tree/main): Great Evaluation Benchmarking Framework.

