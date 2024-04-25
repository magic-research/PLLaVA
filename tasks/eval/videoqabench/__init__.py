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
client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

task_type2chatgpt_contents = OrderedDict({
    "MSVD_QA": {
        "system": "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
                  "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
                  "------"
                  "##INSTRUCTIONS: "
                  "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
                  "- Consider synonyms or paraphrases as valid matches.\n"
                  "- Evaluate the correctness of the prediction compared to the answer.",
        "user": """Please evaluate the following video-based question-answer pair:\n\n"""
                """Question: {question}\n"""
                """Correct Answer: {answer}\n"""
                """Predicted Answer: {pred}\n\n"""
                """Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. """
                """Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."""
                """DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. """
                """For example, your response should look like this: {{'pred': 'yes', 'score': 4.8}}."""
    },
    "MSRVTT_QA": {
        "system": "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
                  "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
                  "------"
                  "##INSTRUCTIONS: "
                  "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
                  "- Consider synonyms or paraphrases as valid matches.\n"
                  "- Evaluate the correctness of the prediction compared to the answer.",
        "user": """Please evaluate the following video-based question-answer pair:\n\n"""
                """Question: {question}\n"""
                """Correct Answer: {answer}\n"""
                """Predicted Answer: {pred}\n\n"""
                """Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. """
                """Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."""
                """DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. """
                """For example, your response should look like this: {{'pred': 'yes', 'score': 4.8}}."""
                # """Make sure you only response with text that Follows Python syntax. For example, your response should look like this: {'pred': 'yes', 'score': 4.8}."""
    },
    "ActivityNet": {
        "system": "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
                  "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
                  "------"
                  "##INSTRUCTIONS: "
                  "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
                  "- Consider synonyms or paraphrases as valid matches.\n"
                  "- Evaluate the correctness of the prediction compared to the answer.",
        "user": """Please evaluate the following video-based question-answer pair:\n\n"""
                """Question: {question}\n"""
                """Correct Answer: {answer}\n"""
                """Predicted Answer: {pred}\n\n"""
                """Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. """
                """Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."""
                """DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. """
                """For example, your response should look like this: {{'pred': 'yes', 'score': 4.8}}."""
                # """Make sure you only response with text that Follows Python syntax. For example, your response should look like this: {'pred': 'yes', 'score': 4.8}."""
    },
    "TGIF_QA": {
        "system": "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
                  "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
                  "------"
                  "##INSTRUCTIONS: "
                  "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
                  "- Consider synonyms or paraphrases as valid matches.\n"
                  "- Evaluate the correctness of the prediction compared to the answer.",
        "user": """Please evaluate the following video-based question-answer pair:\n\n"""
                """Question: {question}\n"""
                """Correct Answer: {answer}\n"""
                """Predicted Answer: {pred}\n\n"""
                """Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. """
                """Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."""
                """DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. """
                """For example, your response should look like this: {{'pred': 'yes', 'score': 4.8}}."""
                # """Make sure you only response with text that Follows Python syntax. For example, your response should look like this: {'pred': 'yes', 'score': 4.8}."""
    },
})

# Follow the instructions carefully and be helpful and precise with your answer.

def check_ans_qa(question, pred, gt, task_type, model="gpt-3.5-turbo-0125"):
    try:
        # Compute the temporal understanding score
        user_input = task_type2chatgpt_contents[task_type]['user']
        user_input = user_input.format(question=question, answer=gt, pred=pred)
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
        response_dict = ast.literal_eval(response_message)
        pred = response_dict['pred']
        score = response_dict['score']
        if not pred in ('yes', 'no') or not isinstance(score, (int, float)):
            raise ValueError(f"{model} doesn't follow")
        flag = pred == 'yes'
    except Exception as e:
        import traceback
        traceback.print_exc()
        flag, score = False, 0
        print(
            f"GPT cannot deal with:\n" 
            f"--pred: {pred}\n"
            f"--gt: {gt}\n"
            f"--gpt responded: {response_message}\n"
            "--will assign flag=False and score=0"
        )
        print(f"Dumb Answer in {task_type}")
    return flag, score

def chatgpt_eval(res, model="gpt-3.5-turbo-0125"):
    pred = res['pred']
    gt = res['gt']
    question=res['question']
    task_type = res['task_type']
    correct, score = check_ans_qa(question=question, pred=pred, gt=gt,task_type=task_type, model=model) # acc is bool, score is given by chatgpt
    # update the scores in result_list for this sample
    res['score'] = score
    res['correct'] = correct
    return res

def save_results(result_list, save_path, model="gpt-3.5-turbo-0125"):
    dump_json(result_list, save_path, 'inference_results.json')
    with Pool(7) as pool:
        func = partial(chatgpt_eval, model=model)
        result_list = [ res for res in tqdm.tqdm(pool.imap_unordered(func, result_list), total=len(result_list), desc='Language Chat Model Automated Evaluation...')]

        # result_list = pool.map(partial(chatgpt_eval, model=model), result_list)
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
    dump_json(all_results, save_path, 'all_results.json')
    dump_json(final_res, save_path, 'upload_leaderboard.json')

def load_results(save_path):
    json_data = load_json(save_path, 'inference_results.json')
    return json_data

@dataclass
class OpenendQASample():
    question: str
    answer: str



class VideoQABenchDataset(EvalDataset):
    data_dir = "DATAS/VideoQA"
    data_list_info = OrderedDict({
        "MSVD_QA": OrderedDict(
            q_json_relpath="MSVD_Zero_Shot_QA/test_q.json", 
            a_json_relpath="MSVD_Zero_Shot_QA/test_a.json", 
            prefix="DATAS/VideoQA/MSVD_Zero_Shot_QA/videos", 
            data_type="video", 
            bound=False,
            question_key='question',
            answer_key='answer',
            name_key='video_name',
            postfix=('avi',),
        ),
        "MSRVTT_QA": OrderedDict(
            q_json_relpath="MSRVTT_Zero_Shot_QA/test_q.json", 
            a_json_relpath="MSRVTT_Zero_Shot_QA/test_a.json", 
            prefix="DATAS/VideoQA/MSRVTT_Zero_Shot_QA/videos/all", 
            data_type="video", 
            bound=False,
            question_key='question',
            answer_key='answer',
            name_key='video_name',
            postfix=('mp4', ),
        ), # don't has start & end 
        "ActivityNet": OrderedDict(
            q_json_relpath="ActivityNet/test_q.json", 
            a_json_relpath="ActivityNet/test_a.json", 
            prefix="DATAS/VideoQA/ActivityNet/all_test", 
            data_type="video", 
            bound=False,
            question_key='question',
            answer_key='answer',
            name_key='video_name',
            postfix=('mp4', 'mkv', 'webm'),
        ), # don't has start & end
        "TGIF_QA": OrderedDict(
            q_json_relpath="TGIF_QA/test_q.json", 
            a_json_relpath="TGIF_QA/test_a.json", 
            prefix="DATAS/VideoQA/TGIF_QA/tgif_videos", 
            data_type="gif", 
            bound=False,
            question_key='question',
            answer_key='answer',
            name_key='video_name',
            postfix=('gif',),
        ), # don't has start & end

    })

    def __init__(self, *args, **kwargs):
        # test_ratio for videoqa is for each sub dataset
        test_ratio = kwargs.pop('test_ratio', None)
        kwargs['test_ratio'] = None
        test_datasets = kwargs.pop('test_datasets', None)
        super().__init__(*args, **kwargs)
        test_ratio = 1 if test_ratio is None else test_ratio
        self.test_ratio = test_ratio
        if test_datasets is not None:
            data_list_info = {k:v for k,v in self.data_list_info.items() if k in test_datasets}
        else:
            data_list_info = self.data_list_info
        data_dir = self.data_dir

        self.data_list = []
        for k, v in data_list_info.items():
            with open(os.path.join(data_dir, v['q_json_relpath']), 'r') as f:
                quesions_json_data = json.load(f)
            with open(os.path.join(data_dir, v['a_json_relpath']), 'r') as f:
                answers_json_data = json.load(f)
            
            indexs = list(range(len(quesions_json_data)))
            np.random.RandomState(42).shuffle(indexs)
            num_samples = int(len(indexs) * self.test_ratio) if 0 < self.test_ratio <= 1 else int(self.test_ratio)
            indexs = indexs[:num_samples]
            for i in indexs:
                question_data = quesions_json_data[i]
                answer_data = answers_json_data[i]
                data = {}
                # why do we have anet's video name not in the original json file???
                if k == "ActivityNet":
                    question_data['video_name'] = 'v_' + question_data['video_name']
                data.update(**question_data)
                data.update(**answer_data)
                self.data_list.append({
                    'task_type': k,
                    'data': data,
                    **v, # all the infos
                })
        print(len(self.data_list))
        
    def __len__(self):
        return len(self.data_list)
    
    
    def __getitem__(self, idx):
        decord_method = self.decord_method[self.data_list[idx]['data_type']]
        bound = None
        if self.data_list[idx]['bound']:
            bound = (
                self.data_list[idx]['data']['start'],
                self.data_list[idx]['data']['end'],
            )
        video_name_key = self.data_list[idx]['name_key']
        video_name = self.data_list[idx]['data'][video_name_key]

        video_postfixs = self.data_list[idx]['postfix']
        video_paths =  []
        for p in video_postfixs:
            video_path = os.path.join(self.data_list[idx]['prefix'], video_name + '.' + p)
            if os.path.exists(video_path):
                video_paths.append(video_path) 
        assert len(video_paths) > 0, f'no video named {video_name}'
        # video_filename = self.data_list[idx]['data'][video_name_key] + video_postfix
        video_path = video_paths[0]
        images_group = decord_method(video_path, bound)

        question_key = self.data_list[idx]['question_key']
        answer_key = self.data_list[idx]['answer_key']
        sample = OpenendQASample(
            question=self.data_list[idx]['data'][question_key],
            answer=self.data_list[idx]['data'][answer_key]
        )
        question, answer = self.qa_template(sample)
            
        return {
            'video_pils': images_group, # some might use the original pils and do their own transforms
            'question': question,
            'video_path': video_path,
            'answer': answer,
            'task_type': self.data_list[idx]['task_type']
        }

    def qa_template(self, data: OpenendQASample):
        answer = data.answer
        question = data.question
        # by far, might use some prompting.
        return question, answer


