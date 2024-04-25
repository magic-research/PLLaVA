from functools import partial
import os
import json
from typing import OrderedDict

import tqdm
import torch
from PIL import Image
import ast
import numpy as np
from multiprocessing import Pool

from decord import VideoReader, cpu

import os
from tasks.eval.eval_utils import (
    dump_json,
    load_json,
    EvalDataset,
)
from dataclasses import dataclass
from openai import OpenAI
from utils.easydict import EasyDict
client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

task_type2chatgpt_contents = OrderedDict({
    "Panda70M": {
        "system": "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for video captioning. "
                  "Your task is to compare the predicted captioning with a provided hint (which is usually a ground truth caption provided by human labor or autmated captioning pipeline)." 
                  "You should determine if they match meaningfully, logically and precisely. Here's how you can accomplish the task:"
                  "------"
                  "##INSTRUCTIONS: "
                  "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
                  "- Consider synonyms or paraphrases as valid matches.\n"
                  "- Evaluate the correctness of the prediction compared to the answer.",
        "user": """Please evaluate the following video-based Captioning pair:\n\n"""
                """Caption: {caption}\n"""
                """Predicted Caption: {pred}\n\n"""
                """Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. """
                """Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."""
                """DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. """
                """For example, your response should look like this: {{'pred': 'yes', 'score': 4.8}}."""
    },
})

# Follow the instructions carefully and be helpful and precise with your answer.

def check_ans_recaption(pred, gt, task_type, model="gpt-3.5-turbo-0125"):
    try:
        # Compute the temporal understanding score
        user_input = task_type2chatgpt_contents[task_type]['user']
        user_input = user_input.format(caption=gt, pred=pred)
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": task_type2chatgpt_contents[task_type]['system'],
                },
                {
                    "role": "user",
                    "content": user_input,
                }
            ]
        )
        # Convert response to a Python dictionary.
        # response_message = completion["choices"][0]["message"]["content"]
        response_message = completion.choices[0].message.content
        num_tokens_openai = completion.usage.total_tokens
        response_dict = ast.literal_eval(response_message)
        pred = response_dict['pred']
        score = response_dict['score']
        if not pred in ('yes', 'no') or not isinstance(score, (int, float)):
            raise ValueError(f"{model} doesn't follow")
        flag = pred == 'yes'
    except Exception as e:
        import traceback
        traceback.print_exc()
        flag, score, num_tokens_openai = False, 0, 0
        print(
            f"GPT cannot deal with:\n" 
            f"--pred: {pred}\n"
            f"--gt: {gt}\n"
            f"--gpt responded: {response_message}\n"
            "--will assign flag=False and score=0"
        )
        print(f"Dumb Answer in {task_type}")
    return flag, score, num_tokens_openai

def chatgpt_eval(res, model="gpt-3.5-turbo-0125"):
    pred = res['pred']
    gt = res['caption']
    task_type = res['task_type']
    correct, score, num_tokens_openai = check_ans_recaption(pred=pred, gt=gt,task_type=task_type, model=model) # acc is bool, score is given by chatgpt
    # update the scores in result_list for this sample
    res['score'] = score
    res['correct'] = correct
    res['num_tokens_openai'] = num_tokens_openai
    return res

def save_results(result_list, save_path, model="gpt-3.5-turbo-0125"):
    dump_json(result_list, save_path, 'inference_results.json')
    with Pool(7) as pool:
        func = partial(chatgpt_eval, model=model)
        result_list = [ res for res in tqdm.tqdm(pool.imap_unordered(func, result_list), total=len(result_list), desc='Language Chat Model Automated Evaluation...')]

    # result_list = [chatgpt_eval(res, model=model) for res in result_list]

    final_res, acc_dict = {}, {}
    correct, total, total_score = 0, 0, 0
    for i, res in enumerate(result_list):
        task_type = res['task_type']
        if task_type not in acc_dict:
            acc_dict[task_type] = {
                'correct': 0,
                'total': 0,
                'score': 0,
            } # correct, total
        acc_dict[task_type]['total'] += 1
        acc_dict[task_type]['correct'] += res['correct']
        acc_dict[task_type]['score'] += res['score']
     
    for k, v in acc_dict.items():
        final_res[k] = {
            'acc': v['correct'] / v['total'] * 100,
            'score': v['score'] / v['total']
        }
        correct += v['correct']
        total += v['total']
        total_score += v['score']

    final_res['Avg_Acc'] = correct / total * 100
    final_res['Avg_Score'] = total_score / total

    all_results = {
        "acc_dict": acc_dict,
        "result_list": result_list
    }
    dump_json(all_results, save_path, f'final_results-{model}.json')
    dump_json(final_res, save_path, 'upload_leaderboard.json')

def load_results(save_path, model="gpt-3.5-turbo-0125"):
    result_list = load_json(save_path, f'final_results-{model}.json')
    if result_list is not None:
        result_list = result_list['result_list']
    
    if result_list is None:
        result_list = load_json(save_path, 'inference_results.json')   

    return result_list

class CaptionSample(EasyDict):
    def get_info(self):
        return {}

class RecaptionSample(EasyDict):
    caption: str
    def get_info(self):
        # template = ("""To facilitate success in the task, I'll offer hints from the automated image captioning pipeline's output on the frames. """
        #             """Please note that this information may contain noise but remains descriptive."""
        #             """Presented below are the noisy details:\n"""
        #             """Hint: {hint}\n"""
        #             """The hint comprises noisy captions generated for certain frames in the video. """
        #             """Please refrain from disclosing the original hints provided; instead, provide rewritten accurate information.""")
        # hint = template.format(hint=self.hint,)
        return {
            "noisy_caption": self.caption
        }

class RecaptionSampleWithMatchingScore(EasyDict):
    caption: str
    matching_score: float

    def get_info(self):
        # template = ("""To facilitate success in the task, I'll offer hints from the automated image captioning pipeline's output on the frames. """
        #             """Please note that this information may contain noise but remains descriptive."""
        #             """Presented below are the noisy details:\n"""
        #             """Hint: {hint}\n"""
        #             """Matching Score: {matching_score:.02f}\n"""
        #             """The hint comprises noisy captions generated for certain frames in the video. """
        #             """Matching scores indicate the likelihood of these captions matching the original frames.\n"""
        #             """Please refrain from disclosing the original hints provided; instead, provide rewritten accurate information."""
        #             )
                    
        # hint = template.format(hint=self.hint,
        #                        matching_score=self.matching_score)
        info = {
            "noisy_caption": self.caption,
            "matching_score": self.matching_score,
        }
        # by far, might use some prompting.
        return info

class RecaptionDataset(EvalDataset):
    data_dir = "DATAS/Recaption"
    data_list_info = OrderedDict({
        # "Panda70M": OrderedDict(
        #     json_relpath="Panda70M/annotations.json", 
        #     prefix="DATAS/Recaption/Panda70M/videos", 
        #     data_type="video", 
        #     bound=False,
        #     key_rename_map={
        #         # 'caption': 'hint',
        #     },
        #     name_key='video_name',
        #     postfix=('mp4', 'mkv', 'webm'),
        #     recaption_type=RecaptionSample,
        # ), # don't has start & end
        "Inter4K": OrderedDict(
            json_relpath="Inter4K/annotations.json", 
            prefix="DATAS/Recaption/Inter4K/60fps/UHD", 
            data_type="video", 
            bound=False,
            key_rename_map={
                # 'caption': 'hint',
            },
            name_key='video_name',
            postfix=('mp4', 'mkv', 'webm'),
            recaption_type=CaptionSample,
        ), # don't has start & end
    })

    def __init__(self, *args, **kwargs):
        # recaption's test_ratio should shuffle the dataset
        test_ratio = kwargs.pop('test_ratio', None)
        super().__init__(*args, **kwargs)
        self.test_ratio = test_ratio
        test_ratio = 1. if test_ratio is None else test_ratio
        data_list_info = self.data_list_info
        data_dir = self.data_dir

        self.data_list = []
        for k, v in data_list_info.items():
            with open(os.path.join(data_dir, v['json_relpath']), 'r') as f:
                annotation_json_data = json.load(f)

            indexs = list(range(len(annotation_json_data)))
            np.random.RandomState(42).shuffle(indexs)
            num_samples = int(len(indexs) * test_ratio) if 0 < test_ratio <= 1 else int(test_ratio)
            indexs = indexs[:num_samples]
            for i in indexs:
                annotation_data = annotation_json_data[i]
                for key_old, key_new in v['key_rename_map'].items():
                    # temporary renameing the keys
                    value = annotation_data.pop(key_old)
                    annotation_data[key_new] = value

                data = dict(annotation_data)
                self.data_list.append({
                    'task_type': k,
                    'data': data,
                })

    def __getitem__(self, idx):
        task_type = self.data_list[idx]['task_type']
        decord_method = self.decord_method[self.data_list_info[task_type]['data_type']]
        bound = None

        if self.data_list_info[task_type]['bound']:
            bound = (
                self.data_list[idx]['data']['start'],
                self.data_list[idx]['data']['end'],
            )
        video_name_key = self.data_list_info[task_type]['name_key']
        video_name = self.data_list[idx]['data'][video_name_key]

        video_postfixs = self.data_list_info[task_type]['postfix']
        video_paths =  []
        for p in video_postfixs:
            video_path = os.path.join(self.data_list_info[task_type]['prefix'], video_name + '.' + p)
            if os.path.exists(video_path):
                video_paths.append(video_path)
        assert len(video_paths) > 0, f'no video named {video_name}'
        # video_filename = self.data_list[idx]['data'][video_name_key] + video_postfix
        video_path = video_paths[0]
        images_group = decord_method(video_path, bound)

        sample = self.data_list_info[task_type]['recaption_type'](**self.data_list[idx]['data'],)
        info = sample.get_info()
            
        return {
            'video_pils': images_group, # some might use the original pils and do their own transforms
            'video_path': video_path,
            'info': info,
            'sample': sample,
            'task_type': task_type,
        }



