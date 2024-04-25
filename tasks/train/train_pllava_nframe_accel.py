import datetime
import gc
import time
import os
import os.path as osp
import re
import itertools
import functools
import random
import math
import shutil
from typing import Optional, Union

import torch
import numpy as np
from safetensors import safe_open

import logging
from accelerate.logging import get_logger
from accelerate import Accelerator, DistributedType
from accelerate.utils import set_seed
from peft import get_peft_model, LoraConfig, TaskType


from dataset import create_dataset, create_loader
from tasks.shared_utils import get_media_types
from utils.basic_utils import (MetricLogger, SmoothedValue, setup_seed)
from utils.config_utils import setup_main
from transformers.utils import TensorType

from tasks.shared_utils import create_optimizer, create_scheduler
import copy
from transformers import  (
    DataCollatorWithPadding,
    get_scheduler,
    AutoModel,
    AutoModelForCausalLM
    )
from models.pllava import PllavaConfig, PllavaForConditionalGeneration, PllavaProcessor

# logger = logging.getLogger(__name__)
IMAGE_TOKEN='<image>'

logger = get_logger(__name__)

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_state_maybe_zero_3(named_params, keys_to_match=["lora_","multi_modal_projector"]):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return

def setup_dataloaders(config, mode="pt", collate_fn=None):
    # train datasets, create a list of data loaders
    logger.info(f"Creating dataset for {mode}")
    train_datasets = create_dataset(f"{mode}_train", config)

    media_types = get_media_types(train_datasets)
    samplers = [None] * len(media_types)

    train_loaders = create_loader(
        train_datasets,
        samplers,
        batch_size=[config.inputs.batch_size[k] for k in media_types],
        num_workers=[config.num_workers] * len(media_types),
        is_trains=[True] * len(media_types),
        collate_fns=[collate_fn] * len(media_types),
    )  # [0]

    return train_loaders, media_types


def setup_model(
    config, find_unused_parameters=False
):
    if config.model.torch_dtype in ('bfloat16', 'float16', 'float32'):
        torch_dtype = eval(f'torch.{config.model.torch_dtype}')
    else:
        torch_dtype = config.model.torch_dtype
    logger.info("Creating model")
    
    processor = PllavaProcessor.from_pretrained(config.model.repo_id, 
                                               padding_side='right', 
                                               center_pad=config.preprocess.center_pad,
                                               )
    

    model_config = PllavaConfig.from_pretrained(config.model.repo_id,
                                               torch_dtype=torch_dtype, 
                                               num_frames=config.model.num_frames,
                                               pooling_method=config.model.pooling_method,
                                               image_token_index=config.preprocess.image_token_index,
                                               frame_shape=config.model.frame_shape,
                                               pooling_shape=config.model.pooling_shape,
                                               use_pooling=config.model.use_pooling,
                                               gradient_checkpointing=config.gradient_checkpointing,
                                               )
    print("====>gradient_checkpointing",model_config.gradient_checkpointing)

    model = PllavaForConditionalGeneration.from_pretrained(config.model.repo_id, config=model_config, torch_dtype=torch_dtype)

    if config.model.load_from_origin:
        with torch.no_grad():
            lm_model = AutoModelForCausalLM.from_pretrained(config.model.origin_llm, torch_dtype=torch_dtype, device_map="cpu",)
        with torch.no_grad():
            clip = AutoModel.from_pretrained(config.model.origin_vision, torch_dtype=torch_dtype, device_map="cpu",)
        msg = model.vision_tower.load_state_dict(clip.state_dict(), strict=False)
        # print(msg)
        msg = model.language_model.load_state_dict(lm_model.state_dict(), strict=False)
        print(msg)

        
    if config.model.freeze_lm:
        logger.info("freezing parameters in model.language_model")
        for p in model.language_model.parameters():
            p.requires_grad = False

    if config.model.freeze_projector:
        logger.info("freezing parameters in model.multi_modal_projector")
        for p in model.multi_modal_projector.parameters():
            p.requires_grad = False

    if config.model.freeze_vision_tower:
        logger.info("freezing parameters in model.vision_tower")
        for p in model.vision_tower.parameters():
            p.requires_grad = False

    if config.model.use_lora:
        logger.info("getting LoRA Language Model")
        kwargs = {}
        if config.model.lora_target_modules is not None and len(config.model.lora_target_modules) > 0:
            kwargs.update({"target_modules": config.model.lora_target_modules})
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False, 
            r=config.model.lora_r, lora_alpha=config.model.lora_alpha, lora_dropout=config.model.lora_dropout,
            **kwargs
        )
        model.language_model = get_peft_model(model.language_model, peft_config)
        model.language_model.print_trainable_parameters()

    if config.model.pretrained_path is not None and not config.deepspeed:
        logger.info("======> loading pretrained weights from " + str(config.model.pretrained_path))
        state_dict = {}
        save_fnames = os.listdir(config.model.pretrained_path)
        if "model.safetensors" in save_fnames:
            print("Loading weight from", config.model.pretrained_path, "model.safetensors")
            with safe_open(f"{config.model.pretrained_path}/model.safetensors", framework="pt", device="cpu") as f:
                for k in f.keys():
                    state_dict[k] = f.get_tensor(k)
        else:
            print("Loading weight from", config.model.pretrained_path)
            for fn in save_fnames:
                if fn.startswith('model-0000'):
                    with safe_open(f"{config.model.pretrained_path}/{fn}", framework="pt", device="cpu") as f:
                        for k in f.keys():
                            state_dict[k] = f.get_tensor(k)
        
        if 'model' in state_dict.keys():
            msg = model.load_state_dict(state_dict['model'], strict=False)
        else:
            msg = model.load_state_dict(state_dict, strict=False)
        logger.info(msg)
        logger.info("=====> Finish loading")

    return model, processor

def setup_optimizer_and_scheduler(config, model):
    optimizer = create_optimizer(config.optimizer, model) # do you want to filter bias and bn?
    if config.scheduler.is_videochat2_custom:
        scheduler = create_scheduler(config.scheduler, optimizer)
    else:
        scheduler=None

    return optimizer, scheduler

class RandomMappingIterator():
    # a random iter through the multiple mapping style dataloaders
    def __init__(self, train_loaders, media_types, resume_step=0):
        self.train_loaders = train_loaders
        self.media_types = media_types
        self.total_num_samples = sum(len(train_loader) for train_loader in self.train_loaders)
        self.weights = [len(loader) / self.total_num_samples for loader in train_loaders]
        self.resume_step = resume_step
        if resume_step != 0:
            self.total_num_samples= self.total_num_samples-resume_step
            # remove corresponding iters from each loader


    def __iter__(self):
        train_loaders = self.train_loaders
        iters = [iter(train_loader) for train_loader in train_loaders]
        
        media_types = copy.deepcopy(self.media_types)
        weights = copy.deepcopy(self.weights)
        while len(iters) > 0:
            index = np.random.choice(list(range(len(iters))), p=weights, replace=True)
            try:
                batch = next(iters[index])
            except StopIteration as e:
                iters.pop(index)
                media_types.pop(index)
                weights.pop(index)
                total = sum(weights)
                weights = [w/total for w in weights]
                continue

            media_type = media_types[index]
            yield media_type, batch

    def __len__(self):
        return self.total_num_samples

def split_and_record_separators(input_string, separators) -> list:
    texts = [input_string]
    for sep in separators:
        new_texts = []
        for text in texts:
            if sep not in text:
                new_texts.append(text)
            else:
                split_strings = text.split(sep)
                joint_strings = [t for pair in zip(split_strings[:-1], itertools.repeat(sep)) for t in pair ] + split_strings[-1:]
                new_texts.extend(joint_strings)
        texts = new_texts
    return texts

def preprocess(
    batch,
    args,
    processor,
    collate_fn,
    dtype=torch.bfloat16,
    return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
):
    tokenizer = processor.tokenizer
    # tokenization for training
    max_length = args.max_txt_l
    input_list, images = [], []
    for sample in batch:
        image, tex, instruction, index = sample  # (nframe, 3, h, w), (0-255)
        num_img = image.shape[0]
        tex = tex.replace(args.dataset_video_placeholder, IMAGE_TOKEN).replace(args.dataset_image_placeholder, IMAGE_TOKEN)
        seps = [role for role in args.roles]
        segs = split_and_record_separators(tex, seps)
        input_ids, labels, attention_mask = [], [], []

        for i, seg in enumerate(segs):
            seg_ignore = False if seg == seps[1] else \
                        (True if  i == 0 or seg in seps else seg_ignore) # not ignoring assistant, changing in sepecific situations
            current_ignore = True if seg in seps else seg_ignore # serve for only this one iteration
            seg_input_ids = tokenizer.encode(seg, add_special_tokens=True if i==0 else False) # only add bos token
            seg_labels = [args.ignore_index] * len(seg_input_ids) if current_ignore else seg_input_ids
            seg_attention_mask = [1] * len(seg_input_ids) # do attend
            input_ids.extend(seg_input_ids)
            labels.extend(seg_labels)
            attention_mask.extend(seg_attention_mask)

        pad_length = max_length - len(input_ids)
        labels = labels[:max_length]
        attention_mask = attention_mask[:max_length]
        input_ids=input_ids[:max_length]

        labels = labels + [args.ignore_index] * pad_length # padding doesn't take care of labels. do the padding here
        input_ids = input_ids + [tokenizer.pad_token_id] * pad_length
        attention_mask =  attention_mask + [0]*pad_length
        sample_input = {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
        }
        input_list.append(sample_input)
        images.append(image if image.ndim==4 else image.unsqueeze(0)) # made 4 dim for image, remain 4 dim for video
    
    inputs = collate_fn(input_list)
    
    # interpolate frames if the total frame is smaller than needed    
    for i, video in enumerate(images):
        if video.shape[0] < args.num_frames:
            multiplier = int(args.num_frames/video.shape[0]) + 1
            video = video.repeat_interleave(multiplier, dim=0)[:args.num_frames]
            images[i] = video
            assert video.shape[0] == args.num_frames
    if args.clip_transform:
        multimodal_features = processor(images=images)
        inputs.update(**multimodal_features)
    else:
        inputs["pixel_values"] = torch.concat(images) # already processed to features in dataset get item


    return inputs

def main(config):
    accelerator_log_kwargs=dict(
        log_with=config.report_to,
        project_dir=config.output_dir
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        **accelerator_log_kwargs
    )
    logger.info(f"train_file: {config.train_file}")
    model, processor = setup_model(
        config,
        find_unused_parameters=True,
    )
    if accelerator.is_main_process:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)
    
    collate_fn = DataCollatorWithPadding(tokenizer=processor.tokenizer, padding='max_length', max_length=config.max_txt_l, return_tensors='pt',)
    collate_fn = functools.partial(preprocess, args=config.preprocess, processor=processor, collate_fn=collate_fn)
    train_loaders, train_media_types = setup_dataloaders(config, mode=config.mode, collate_fn=collate_fn)
    num_steps_per_epoch = math.ceil(sum(len(d) for d in train_loaders) / config.gradient_accumulation_steps)
    # load optimizer and custom scheduler
    config.scheduler.num_training_steps = num_steps_per_epoch * config.scheduler.epochs
    config.scheduler.num_warmup_steps = math.ceil(config.scheduler.num_training_steps * config.scheduler.warmup_ratio)
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(config, model) 
    # if not set customized scheduler, default hf scheduler
    overrode_max_train_steps = False
    if config.max_train_steps is None:
        config.max_train_steps = config.scheduler.epochs * num_steps_per_epoch
        overrode_max_train_steps = True
    if lr_scheduler is None:
        lr_scheduler = get_scheduler(
            name=config.scheduler.sched,
            optimizer=optimizer,
            num_warmup_steps=config.scheduler.num_warmup_steps,
            num_training_steps=config.max_train_steps
            if overrode_max_train_steps
            else config.max_train_steps * accelerator.num_processes,
        )
    model, optimizer, lr_scheduler, *train_loaders = accelerator.prepare(
        model, optimizer, lr_scheduler, *train_loaders
    )

    if hasattr(config, 'seed'):
        set_seed(config.seed)
    
    experiment_config = { # include all the important hyperparam
        'num_frames': config.num_frames, 
        'max_txt_l': config.max_txt_l, 
        'batch_size': config.batch_size, 
    }

    model.train()

    start_epoch = 0
    num_batches = sum(len(loader) for loader in train_loaders)
    global_step = start_epoch * num_batches  # the steps before divided by accumulation
    if osp.exists(config.output_dir):
        subfolders = os.listdir(config.output_dir)
        sample_saving = False
        for subfolder in subfolders:
            if subfolder.endswith("M"):
                sample_saving = True
        if sample_saving:
            ckpt_paths = [subfolder for subfolder in subfolders if re.match(r'ckpt_resume_[\d.]+M$', subfolder) is not None]
            ckpt_iters = [float(re.findall(r'[\d.]+', x)[0]) for x in ckpt_paths]
        else:
            ckpt_paths = [subfolder for subfolder in subfolders if re.match("ckpt_[^\d]+", subfolder) is not None]
            ckpt_iters = [int(s.split(re.match("ckpt_[^\d]+", s).group())[-1]) for s in ckpt_paths]

    
        resume_cur_epoch_step=0
        if len(ckpt_iters) > 0:
            resume_iter = max(ckpt_iters)
            ckpt_path = osp.join(config.output_dir, ckpt_paths[ckpt_iters.index(resume_iter)])
            accelerator.print(f"Resumed from checkpoint: {ckpt_path}")
            accelerator.load_state(ckpt_path)
            if sample_saving:
                resume_iter = int(resume_iter*1e6/(config.batch_size*accelerator.state.num_processes))

            if "epoch" in ckpt_path:
                start_epoch = int(resume_iter) + 1
                resume_cur_epoch_step = 0
                global_step = start_epoch * num_batches
            else:
                # need to multiply `gradient_accumulation_steps` to reflect real steps
                # num_finish_smaple = int(max_ckpt_num) * config.gradient_accumulation_steps
                start_epoch = resume_iter // num_batches
                global_step = resume_iter 
                resume_cur_epoch_step = resume_iter - start_epoch * num_batches
            accelerator.print(f"Resume from epoch {start_epoch}, steps{resume_cur_epoch_step}")
            
    

    # TensorBoard cannot log Enums, need the raw value
    accelerator.init_trackers("train_pllava_nframe", experiment_config)
    start_time = time.time()
    


    logger.info(f"Start training {str(start_time)}, from start_epoch-{start_epoch}, step-{resume_cur_epoch_step}")

    # skip the first `n` batches in the dataloader when resuming from a checkpoint
    active_train_loaders = train_loaders 
    if resume_cur_epoch_step > 0:
        active_train_loaders = []
        total_dta_num = sum(len(train_loader) for train_loader in train_loaders)
        for train_loader in train_loaders:
            skip_batch_num = int((resume_cur_epoch_step/total_dta_num)*len(train_loader))
            skipped_train_loader = accelerator.skip_first_batches(train_loader, num_batches=skip_batch_num)
            active_train_loaders.append(skipped_train_loader)
    
    media_types = get_media_types(active_train_loaders)
    train_loader = RandomMappingIterator(active_train_loaders, media_types)

    for epoch in range(start_epoch, config.scheduler.epochs):  
        if not config.evaluate:
            gc.collect()
            torch.cuda.empty_cache()
            metric_logger = MetricLogger(delimiter="  ")
            loss_names = ["loss"]
            for name in loss_names:
                for m in media_types:
                    metric_logger.add_meter(
                        f"{m}-{name}", SmoothedValue(window=config.metric_window_size, fmt="{value:.4f}")
                    )

            header = f"Train Epoch: [{epoch}]"
            log_freq = config.log_freq

            iterator = metric_logger.log_every(train_loader, log_freq, header)
            mini_batch_losses = []

            for i, (media_type, inputs) in enumerate(iterator): # video/image, conversation, instruction, index
                    
                with accelerator.accumulate(model):
                    
                    inputs['media_type'] = media_type
                    response = model(**inputs)
                    loss = response.loss
                    mini_batch_losses.append(loss.detach().item())
                    optimizer.zero_grad()
                    accelerator.backward(loss)
                    if config.optimizer.max_grad_norm > 0:
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(model.parameters(), config.optimizer.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                # # logging
                for name in loss_names:
                    value = loss
                    value = value if isinstance(value, float) else value.item()
                    metric_logger.update(**{f"{media_type}-{name}": value})
                global_step += 1
                resume_num_samples = global_step * config.batch_size * accelerator.state.num_processes/1e6

                # save small global step checkpoint in case of breakdown
                if global_step % config.ckpt_steps == 0:
                    accelerator.save_state(output_dir=osp.join(config.output_dir, f"ckpt_resume_{resume_num_samples:.4f}M"))
                    if accelerator.is_main_process:
                        for fn in os.listdir(config.output_dir):
                            if "resume" in fn and fn != f"ckpt_resume_{resume_num_samples:.4f}M":
                                shutil.rmtree(osp.join(config.output_dir, fn))
                
                if global_step % config.save_steps == 0:
                    logger.info(f"global_step {global_step}")
                    with torch.no_grad():
                        accelerator.wait_for_everyone()
                        unwrapped_model = accelerator.unwrap_model(model)
                        if not config.deepspeed:
                            save_state_dict = {k:v for k,v in accelerator.get_state_dict(model).items() if "lora_" in k or "multi_modal_projector" in k}
                        else:
                            save_state_dict = accelerator.get_state_dict(model)
                        unwrapped_model.save_pretrained(osp.join(config.output_dir, f"pretrained_step{resume_num_samples:.4f}M"),
                                            is_main_process=accelerator.is_main_process,
                                            save_function=accelerator.save,
                                            state_dict=save_state_dict)
                        processor.save_pretrained(osp.join(config.output_dir, f"pretrained_step{resume_num_samples:.4f}M"))

                if global_step % log_freq == 0:
                    logs = metric_logger.get_global_avg_dict()
                    logs.update({
                        "step_loss_no_smoothing": accelerator.gather_for_metrics(loss).mean().item(),
                        "epoch": epoch,
                        "step": global_step,
                        "lr": lr_scheduler.get_last_lr()[0],
                    })
                    accelerator.log(logs, step=global_step,)
                    if accelerator.sync_gradients:
                        mini_batch_loss = torch.tensor(mini_batch_losses, device='cuda')
                        accelerator.log({"mini_batch_loss": accelerator.gather_for_metrics(mini_batch_loss).mean().item()},
                                    step=global_step)
                        mini_batch_losses = []
                    

                if config.debug and global_step % 20 == 0:
                    logger.info("debug mode, break training loop")
                    break

                if config.debug and global_step % (2 * log_freq + 3) == 0:
                    logger.info("debug mode, break training loop")
                    break

            # gather the stats from all processes
            metric_logger.synchronize_between_processes()
            logger.info(f"Averaged stats: {metric_logger.global_avg()}")
        logger.info(f"Epoch {epoch}")
        with torch.no_grad():
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            if not config.deepspeed:
                save_state_dict = {k:v for k,v in accelerator.get_state_dict(model).items() if "lora_" in k or "multi_modal_projector" in k}
            else:
                save_state_dict = accelerator.get_state_dict(model)
            unwrapped_model.save_pretrained(osp.join(config.output_dir, f"pretrained_epoch{epoch:02d}"),
                                            is_main_process=accelerator.is_main_process,
                                            save_function=accelerator.save,
                                            state_dict=save_state_dict)
            processor.save_pretrained(osp.join(config.output_dir, f"pretrained_step{epoch:02d}"))
            accelerator.save_state(output_dir=osp.join(config.output_dir, f"ckpt_epoch{epoch:02d}"))


        if config.evaluate:
            break

    accelerator.end_training()
    accelerator.wait_for_everyone()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Training time {total_time_str}")
    logger.info(f"Checkpoints and Logs saved at {config.output_dir}")



if __name__ == "__main__":
    cfg = setup_main()
    print(cfg)
    main(cfg)
