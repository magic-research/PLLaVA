
import functools
import itertools
import json
import logging
from tqdm import tqdm
from PIL import Image
from multiprocessing import Pool
from argparse import ArgumentParser
import multiprocessing as mp



import numpy as np
import torch

import torchvision

import transformers
from decord import VideoReader, cpu

from tasks.eval.model_utils import load_plava, plava_answer
from tasks.eval.eval_utils import ZeroshotDectionHelper, conv_templates

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


IMAGE_TOKEN='<image>'
from tasks.eval.recaption import (
    RecaptionDataset,
    load_results,
    save_results,
)
RESOLUTION = 672 # 

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        default='llava-hf/llava-1.5-7b-hf'
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        default='"./test_results/test_llava_mvbench"'
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        required=True,
        default=4,
    )
    parser.add_argument(
        "--use_lora",
        action='store_true'
    )
    parser.add_argument(
        "--weight_dir",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--eval_model",
        type=str,
        required=False,
        default="gpt-3.5-turbo-0125",
    )
    parser.add_argument(
        '--test_ratio',
        type=float,
        required=False,
        default=None
    )
    parser.add_argument(
        "--conv_mode", 
        type=str,
        required=False,
        default='eval_videoqabench',
    )
    args = parser.parse_args()
    return args

def load_model_and_dataset(rank, world_size, pretrained_model_name_or_path, num_frames, use_lora, weight_dir, test_ratio):
    # remind that, once the model goes larger (30B+) may cause the memory to be heavily used up. Even Tearing Nodes.
    model, processor = load_plava(pretrained_model_name_or_path, num_frames=num_frames, use_lora=use_lora, weight_dir=weight_dir)
    logger.info('done loading llava')
    #  position embedding
    model = model.to(torch.device(rank))
    model = model.eval()
    helper = ZeroshotDectionHelper(torch.device(rank))
    dataset = RecaptionDataset(num_segments=num_frames, test_ratio=test_ratio)
    dataset.set_rank_and_world_size(rank, world_size)
    return model, processor, dataset, helper

def infer_recaption_with_helper(
        model,
        processor,
        helper: ZeroshotDectionHelper,
        data_sample, 
        conv_mode,
        pre_query_prompt=None, # add in the head of question
        post_query_prompt=None, # add in the end of question
        answer_prompt=None, # add in the begining of answer
        return_prompt=None,  # add in the begining of return message
        print_res=False,
    ):
    video_list = data_sample["video_pils"]
    conv = conv_templates[conv_mode].copy()
    info = data_sample['info']
    info.update(helper.help([video_list]))
    info_str = json.dumps(info, indent=4)

    query = ("You are to assist me in accomplishing a task about the input video. Reply to me with a precise yet detailed response. For how you would succeed in the task, read the following sections and then make your response with a elaborate paragraph.\n"
             "# Instructions\n"
             "1. Reason about some questions about the overall video (For example, including but not limited to \"what's going on\", \"where it might be at.\", \"the main sense that the video is conveying\")\n"
             "2. Do not take the provided information as granted. Instead, you should carefully examine the information and reason with the content in the provided video.\n"
             "3. Completely ignore irrelevant information as the detection results and hints are noisy. If you are not sure about something, do not include it in you response. Avoid providing over detailed information such as color, counts of any objects as you are terrible regarding observing these details Also, avoid simply copying/repeating any information as the extra information is not reliable. \n"
             "# Content\n"
             f"```JSON\n{info_str}\n```\n"
             "# Task\n"
             "Describe the background, characters and the actions in the provided video.")

    # query = ("You are to assist me in accomplishing a task about the input video. I will provide you with some extra information as a reference. "
    #          "However, the information would be noisy and have certain faults in it. "
    #          "Your job is to distinguish them from right to wrong and reply to me with a precise yet detailed response. "
    #          "For how you would succeed in the recaptioning task, read the following Instructions section and examine the information in the Content section. "
    #          "Finally, make your response with a elaborate paragraph.\n"
    #          "# Instructions\n"
    #          "1. There might be faults in the provided information. Do not take the provided information as granted.\n"
    #          "2. Instead, you should carefully examine the information and reason with the content in the provided video.\n"
    #          "3. Completely ignore irrelevant information as the detection results and hints are noisy.\n"
    #          "4. Avoid simply copying any information. Carefully make up your response in a narrative tone that is precise yet detailed.\n"
    #          "# Content\n"
    #          f"```JSON\n{info_str}\n```\n"
    #          "# Task\n"
    #          "Describe the background, characters and the actions in the provided video.")
    conv.user_query(query, pre_query_prompt, post_query_prompt, is_mm=True)
    if answer_prompt is not None:
        conv.assistant_response(answer_prompt)

    llm_message, conv = plava_answer(
        conv=conv,
        model=model,
        processor=processor,
        img_list=video_list,
        max_new_tokens=400,
        num_beams=1,
        do_sample=False,
        print_res=print_res
    )
    
    if answer_prompt is not None:
        llm_message =  ''.join(llm_message.split(answer_prompt)[1:])

    if return_prompt is not None:
        llm_message = return_prompt + llm_message

    return llm_message, query
   
def single_test(model, processor, vid_path, num_frames=4, conv_mode="video-chatgpt_v1"):
    def get_index(num_frames, num_segments):
        seg_size = float(num_frames - 1) / num_segments
        start = int(seg_size / 2)
        offsets = np.array([
            start + int(np.round(seg_size * idx)) for idx in range(num_segments)
        ])
        return offsets

    def load_video(video_path, num_segments=8, return_msg=False, num_frames=4, resolution=336):
        transforms = torchvision.transforms.Resize(size=resolution)
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        num_frames = len(vr)
        frame_indices = get_index(num_frames, num_segments)
        images_group = list()
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy())
            images_group.append(transforms(img))
        if return_msg:
            fps = float(vr.get_avg_fps())
            sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
            # " " should be added in the start and end
            msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."
            return images_group, msg
        else:
            return images_group

    if num_frames != 0:
        vid, msg = load_video(vid_path, num_segments=num_frames, return_msg=True, resolution=RESOLUTION)
    else:
        vid, msg = None, 'num_frames is 0, not inputing image'
    img_list = vid

    conv = conv_templates[conv_mode].copy()
    conv.user_query("Describe the video in details.", is_mm=True)
    llm_response, conv = plava_answer(conv=conv, model=model, processor=processor, do_sample=False, img_list=img_list, max_new_tokens=256, print_res=True)

def run(rank, args, world_size):
    if rank != 0:
        transformers.utils.logging.set_verbosity_error()
        logger.setLevel(transformers.logging.ERROR)

    print_res = True
    conv_mode= args.conv_mode
    pre_query_prompt = None
    post_query_prompt = None

    logger.info(f'loading model and constructing dataset to gpu {rank}...')
    model, processor, dataset, helper = load_model_and_dataset(rank,
                                                       world_size,
                                                       pretrained_model_name_or_path=args.pretrained_model_name_or_path,
                                                       num_frames=args.num_frames,
                                                       use_lora=args.use_lora,
                                                       weight_dir=args.weight_dir,
                                                       test_ratio=args.test_ratio)
    logger.info(f'done model and dataset...')
    logger.info('constructing dataset...')
    logger.info('single test...')
    vid_path = "./example/yoga.mp4"
    # vid_path = "./example/jesse_dance.mp4"
    if rank == 0:
        single_test(model, processor, vid_path, num_frames=args.num_frames)
        logger.info('single test done...')
        tbar = tqdm(total=len(dataset))
    logger.info('single test...')

    result_list = []
    done_count = 0
    for example in dataset:
        task_type = example['task_type']
        if task_type in dataset.data_list_info:
            pred, query = infer_recaption_with_helper(
                model,
                processor,
                helper,
                example,
                conv_mode=conv_mode,
                pre_query_prompt=pre_query_prompt,
                post_query_prompt=post_query_prompt,
                print_res=print_res,
            )

            infos = {k: v for k, v in example['sample'].items() if isinstance(v, (str, float, int))}
            res = {
                'pred': pred,
                'task_type': task_type,
                'video_path': example['video_path'],
                'query': query,
                **infos    
            }
        else:
            raise NotImplementedError(f'not implemented task type {task_type}')
        # res = chatgpt_eval(res)
        result_list.append(res)
        if rank == 0:
            tbar.update(len(result_list) - done_count, )
            tbar.set_description_str(
                f"One Chunk--Task Type: {task_type}-"
                f"pred: {pred[:min(15, len(pred))]}......"
            )
            done_count = len(result_list)
    return result_list

def main():
    multiprocess=True
    mp.set_start_method('spawn')
    args = parse_args()
    save_path = args.save_path
    eval_model = args.eval_model
    logger.info(f'trying loading results from {save_path}')
    result_list = load_results(save_path, model=args.eval_model)
    
    if result_list is None:
        if multiprocess:

            logger.info(f'started benchmarking, saving to: {save_path}')
            n_gpus = torch.cuda.device_count()
            # assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
            world_size = n_gpus
            with Pool(world_size) as pool:
                func = functools.partial(run, args=args, world_size=world_size)
                # func = functools.partial(run, world_size=world_size, model=model, dataset=dataset, result_list=[], acc_dict={})
                result_lists = pool.map(func, range(world_size))
            
            logger.info('finished running')

            result_list = [ res for res in itertools.chain(*result_lists)]
        else:
            result_list = run(0, world_size=1, args=args) # debug
    else:
        logger.info(f'loaded results from {save_path}')

    save_results(result_list, save_path, model=eval_model)
    
    
if __name__ == "__main__":
    main()