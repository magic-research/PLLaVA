
import functools
import itertools
import logging
from tqdm import tqdm
from PIL import Image
from multiprocessing import Pool
import multiprocessing as mp
from argparse import ArgumentParser
import numpy as np

import torch
import torchvision

from decord import VideoReader, cpu
import transformers


from tasks.eval.model_utils import load_pllava, pllava_answer
from tasks.eval.eval_utils import conv_templates
from tasks.eval.vcgbench import (
    VideoChatGPTBenchDataset,
    save_results,
    load_results,
)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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
        "--lora_alpha",
        type=int,
        required=False,
        default=32,
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
        "--conv_mode", 
        type=str,
        required=False,
        default='eval_vcgbench',
    )
    parser.add_argument(
        "--test_ratio",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--pooling_shape", 
        type=str,
        required=False,
        default=None,
    )
    args = parser.parse_args()
    return args

def load_model_and_dataset(rank, world_size, pretrained_model_name_or_path, num_frames, use_lora, lora_alpha, weight_dir, test_ratio, pooling_shape=(16,12,12)):
    # remind that, once the model goes larger (30B+) may cause the memory to be heavily used up. Even Tearing Nodes.,
    model, processor = load_pllava(pretrained_model_name_or_path,
                                   num_frames=num_frames,
                                   weight_dir=weight_dir,
                                   use_lora=use_lora,
                                   lora_alpha=lora_alpha,
                                   pooling_shape=pooling_shape)
    logger.info('done loading llava')
    #  position embedding
    model = model.to(torch.device(rank))
    model = model.eval()

    dataset = VideoChatGPTBenchDataset(num_segments=num_frames, test_ratio=test_ratio)
    dataset.set_rank_and_world_size(rank, world_size)
    return model, processor, dataset

def infer_vcgbench(
        model,
        processor,
        data_sample,
        conv_mode,
        pre_query_prompt=None, # add in the head of question
        post_query_prompt=None, # add in the end of question
        print_res=False,
    ):
    video_list = data_sample["video_pils"]
    conv = conv_templates[conv_mode].copy()
    conv.user_query(data_sample['question'], pre_query_prompt, post_query_prompt, is_mm=True)
    stop_criteria_keywords=["###","USER"]

    llm_message, conv = pllava_answer(
        conv=conv,
        model=model,
        processor=processor,
        img_list=video_list,
        max_new_tokens=512,
        do_sample=False,
        print_res=print_res,
        stop_criteria_keywords=stop_criteria_keywords
    )
    

    return llm_message
    
def single_test(model, processor, vid_path, num_frames=4, conv_mode="plain"):
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
    llm_response, conv = pllava_answer(conv=conv, model=model, processor=processor, do_sample=False, img_list=img_list, max_new_tokens=256, print_res=True)

def run(rank, args, world_size,start_rank=0):
    if rank != 0:
        transformers.utils.logging.set_verbosity_error()
        logger.setLevel(transformers.logging.ERROR)
    print_res = True
    conv_mode= args.conv_mode
    pre_query_prompt = None
    post_query_prompt = None
    

    logger.info(f"CONV_MODE: {conv_mode}")

    logger.info(f'loading model and constructing dataset to gpu {rank}...')
    if args.pooling_shape is not None:
        pooling_shape=tuple([int(x) for x in args.pooling_shape.split("-")])
    model, processor, dataset = load_model_and_dataset(rank,
                                                       world_size,
                                                       pretrained_model_name_or_path=args.pretrained_model_name_or_path,
                                                       num_frames=args.num_frames,
                                                       use_lora=args.use_lora,
                                                       weight_dir=args.weight_dir,
                                                       lora_alpha=args.lora_alpha,
                                                       test_ratio=args.test_ratio,
                                                       pooling_shape=pooling_shape)
    logger.info(f'done model and dataset...')
    logger.info('constructing dataset...')
    logger.info('single test...')
    vid_path = "./example/yoga.mp4"
    if rank == 0:
        single_test(model,
                    processor,
                    vid_path,
                    num_frames=args.num_frames,
                    conv_mode=args.conv_mode)
        logger.info('single test done...')
        tbar = tqdm(total=len(dataset))

    result_list = []
    done_count = 0
    for example in dataset:
        task_type = example['task_type']
        gt = example['answer']
        if task_type == 'consistency_qa':
            assert 'question' in example and 'question1' in example, 'two questions'
            pred = infer_vcgbench(
                model,
                processor,
                example, 
                conv_mode=conv_mode,
                pre_query_prompt=pre_query_prompt,
                post_query_prompt=post_query_prompt,
                print_res=print_res,
            )
            # inference the other question
            example['question'], example['question1'] = example['question1'], example['question']
            pred1 = infer_vcgbench(
                model,
                processor,
                example, 
                conv_mode=conv_mode,
                pre_query_prompt=pre_query_prompt,
                post_query_prompt=post_query_prompt,
                print_res=print_res,
            )
            res = {
                'pred': pred,
                'pred1': pred1,
                'gt': gt,
                'video': example['video_path'],
                'task_type': task_type,
                'question': example['question'],
                'question1': example['question1'],
            }
        elif task_type in dataset.data_list_info:
            pred = infer_vcgbench(
                model,
                processor,
                example, 
                conv_mode=conv_mode,
                pre_query_prompt=pre_query_prompt,
                post_query_prompt=post_query_prompt,
                print_res=print_res,
            )
            res = {
                'pred': pred,
                'gt': gt,
                'video_path': example['video_path'],
                'question': example['question'],
                'task_type': task_type,
            }
        else:
            raise NotImplementedError(f'not implemented task type {task_type}')

        result_list.append(res)
        if rank == 0:
            tbar.update(len(result_list) - done_count, )
            tbar.set_description_str(
                f"One Chunk--Task Type: {task_type}-"
                f"gt: {gt[:min(15, len(gt))]}......--pred: {pred[:min(15, len(gt))]}......"
            )
            done_count = len(result_list)
    return result_list

def main():
    multiprocess=True
    mp.set_start_method('spawn')
    args = parse_args()
    save_path = args.save_path
    eval_model = args.eval_model
    result_list = load_results(save_path)
    start_rank=0

    if result_list is None:
        if multiprocess:
            logger.info(f'started benchmarking, saving to: {save_path}')
            n_gpus = torch.cuda.device_count()
            # assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
            world_size = n_gpus
            with Pool(world_size) as pool:
                func = functools.partial(run, args=args, world_size=world_size, start_rank=start_rank)
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