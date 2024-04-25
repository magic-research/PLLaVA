import ast
import os
import json
from typing import OrderedDict
from multiprocessing import Pool
from functools import partial

import tqdm

from tasks.eval.eval_utils import (
    dump_json,
    load_json,
    EvalDataset,
)

from openai import OpenAI
client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

sub_task_type2chatgpt_contents = OrderedDict({
    # general ones
    'temporal': {
        "system": "You are an intelligent chatbot designed for evaluating the temporal understanding of generative outputs for video-based question-answer pairs. "
                  "Your task is to compare the predicted answer with the correct answer and determine if they correctly reflect the temporal sequence of events in the video content. Here's how you can accomplish the task:"
                  "------"
                  "##INSTRUCTIONS: "
                  "- Focus on the temporal consistency between the predicted answer and the correct answer. The predicted answer should correctly reflect the sequence of events or details as they are presented in the video content.\n"
                  "- Consider synonyms or paraphrases as valid matches, but only if the temporal order is maintained.\n"
                  "- Evaluate the temporal accuracy of the prediction compared to the answer.",
        "user": "Please evaluate the following video-based question-answer pair:\n\n"
                "Question: {question}\n"
                "Correct Answer: {answer}\n"
                "Predicted Answer: {pred}\n\n"
                "Provide your evaluation only as a temporal accuracy score where the temporal accuracy score is an integer value between 0 and 5, with 5 indicating the highest level of temporal consistency. "
                "Please generate the response in the form of a Python dictionary string with keys 'score', where its value is the temporal accuracy score in INTEGER, not STRING."
                "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                "For example, your response should look like this: {{'score': 4.8}}."        
    },
    "context": {
        "system": "You are an intelligent chatbot designed for evaluating the contextual understanding of generative outputs for video-based question-answer pairs. "
                  "Your task is to compare the predicted answer with the correct answer and determine if the generated response aligns with the overall context of the video content. Here's how you can accomplish the task:"
                  "------"
                  "##INSTRUCTIONS: "
                  "- Evaluate whether the predicted answer aligns with the overall context of the video content. It should not provide information that is out of context or misaligned.\n"
                  "- The predicted answer must capture the main themes and sentiments of the video.\n"
                  "- Consider synonyms or paraphrases as valid matches.\n"
                  "- Provide your evaluation of the contextual understanding of the prediction compared to the answer.",
        "user": "Please evaluate the following video-based question-answer pair:\n\n"
                "Question: {question}\n"
                "Correct Answer: {answer}\n"
                "Predicted Answer: {pred}\n\n"
                "Provide your evaluation only as a contextual understanding score where the contextual understanding score is an integer value between 0 and 5, with 5 indicating the highest level of contextual understanding. "
                "Please generate the response in the form of a Python dictionary string with keys 'score', where its value is contextual understanding score in INTEGER, not STRING."
                "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                "For example, your response should look like this: {{'score': 4.8}}."
    },
    'detailed_orientation': {
        "system": "You are an intelligent chatbot designed for evaluating the detail orientation of generative outputs for video-based question-answer pairs. "
                  "Your task is to compare the predicted answer with the correct answer and determine its level of detail, considering both completeness and specificity. Here's how you can accomplish the task:"
                  "------"
                  "##INSTRUCTIONS: "
                  "- Check if the predicted answer covers all major points from the video. The response should not leave out any key aspects.\n"
                  "- Evaluate whether the predicted answer includes specific details rather than just generic points. It should provide comprehensive information that is tied to specific elements of the video.\n"
                  "- Consider synonyms or paraphrases as valid matches.\n"
                  "- Provide a single evaluation score that reflects the level of detail orientation of the prediction, considering both completeness and specificity.",
        "user": "Please evaluate the following video-based question-answer pair:\n\n"
                "Question: {question}\n"
                "Correct Answer: {answer}\n"
                "Predicted Answer: {pred}\n\n"
                "Provide your evaluation only as a detail orientation score where the detail orientation score is an integer value between 0 and 5, with 5 indicating the highest level of detail orientation. "
                "Please generate the response in the form of a Python dictionary string with keys 'score', where its value is the detail orientation score in INTEGER, not STRING."
                "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                "For example, your response should look like this: {{'score': 4.8}}."
        ,
    },
    "correctness": {
        "system": "You are an intelligent chatbot designed for evaluating the factual accuracy of generative outputs for video-based question-answer pairs. "
                   "Your task is to compare the predicted answer with the correct answer and determine if they are factually consistent. Here's how you can accomplish the task:"
                   "------"
                   "##INSTRUCTIONS: "
                   "- Focus on the factual consistency between the predicted answer and the correct answer. The predicted answer should not contain any misinterpretations or misinformation.\n"
                   "- The predicted answer must be factually accurate and align with the video content.\n"
                   "- Consider synonyms or paraphrases as valid matches.\n"
                   "- Evaluate the factual accuracy of the prediction compared to the answer.",
        "user": "Please evaluate the following video-based question-answer pair:\n\n"
                "Question: {question}\n"
                "Correct Answer: {answer}\n"
                "Predicted Answer: {pred}\n\n"
                "Provide your evaluation only as a factual accuracy score where the factual accuracy score is an integer value between 0 and 5, with 5 indicating the highest level of factual consistency. "
                "Please generate the response in the form of a Python dictionary string with keys 'score', where its value is the factual accuracy score in INTEGER, not STRING."
                "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                "For example, your response should look like this: {{'score': 4.8}}."
                    
    },
    "consistency": {
        "system": "You are an intelligent chatbot designed for evaluating the consistency of generative outputs for similar video-based question-answer pairs. "
                  "You will be given two very similar questions, a common answer common to both the questions and predicted answers for the two questions ."
                  "Your task is to compare the predicted answers for two very similar question, with a common correct answer and determine if they are consistent. Here's how you can accomplish the task:"
                  "------"
                  "##INSTRUCTIONS: "
                  "- Focus on the consistency between the two predicted answers and the correct answer. Both predicted answers should correspond to the correct answer and to each other, and should not contain any contradictions or significant differences in the conveyed information.\n"
                  "- Both predicted answers must be consistent with each other and the correct answer, in terms of the information they provide about the video content.\n"
                  "- Consider synonyms or paraphrases as valid matches, but only if they maintain the consistency in the conveyed information.\n"
                  "- Evaluate the consistency of the two predicted answers compared to the correct answer.",
        "user":"Please evaluate the following video-based question-answer pair:\n\n"
                "Question 1: {question}\n"
                "Question 2: {question1}\n"
                "Correct Answer: {answer}\n"
                "Predicted Answer to Question 1: {pred}\n"
                "Predicted Answer to Question 2: {pred1}\n\n"
                "Provide your evaluation only as a consistency score where the consistency score is an integer value between 0 and 5, with 5 indicating the highest level of consistency. "
                "Please generate the response in the form of a Python dictionary string with keys 'score', where its value is the consistency score in INTEGER, not STRING."
                "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                "For example, your response should look like this: {{'score': 4.8}}."
                    
    },
})

SYSTEM_VCGBENCH="""
You are Video-ChatGPT, a large vision-language assistant. 
You are able to understand the video content that the user provides, and assist the user with a variety of tasks using natural language.
Follow the instructions carefully and explain your answers in detail based on the provided video.
"""

def check_ans(gt, pred, question, sub_task_type, question1=None, pred1=None, model="gpt-3.5-turbo-0125"):
    # # dummy
    # print('-' * 10 + f'pred: {pred}')
    # print('-' * 10 + f'gt: {gt}')
    try:
        # Compute the temporal understanding score
        user_input = sub_task_type2chatgpt_contents[sub_task_type]['user']
        if question1 is not None and pred1 is not None:
            assert sub_task_type == 'consistency', 'consistency has two answers'
            user_input = user_input.format(question=question, answer=gt, pred=pred, pred1=pred1, question1=question1)
        else:
            user_input = user_input.format(question=question, answer=gt, pred=pred)
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": sub_task_type2chatgpt_contents[sub_task_type]['system'],
                },
                {
                    "role": "user",
                    "content": user_input,
                }
            ]
        )
        # Convert response to a Python dictionary.
        response_message = completion.choices[0].message.content
        response_dict = ast.literal_eval(response_message)
        flag, score = response_dict['score'] > 3, response_dict['score']
    except Exception as e:
        import traceback
        traceback.print_exc()
        flag, score = False, 0
        print(
            f"GPT cannot deal with:\n" 
            f"--pred: {pred},\n"
            f"--gt: {gt}\n"
            f"--gpt responded: {response_message}\n"
            "--will assign flag=False and score=0"
        )
        print(f"Dumb Answer in {sub_task_type}")
    return flag, score

def chatgpt_eval(res, model="gpt-3.5-turbo-0125"):
    pred = res['pred']
    gt = res['gt']
    question=res['question']
    task_type = res['task_type']
    if task_type == 'generic_qa':
        # eval three sub tasks for generic
        for sub_task_type in ('context', 'detailed_orientation', 'correctness'):
            if pred=="":
                print("no pred")
                score = 0
            else:
                acc, score = check_ans(gt=gt, pred=pred, question=question, sub_task_type=sub_task_type, model=model) # acc is bool, score is given by chatgpt
            # update the scores in result_list for this sample
            res['scores'] = res.get('scores', {})
            res['scores'][sub_task_type] = score
    elif task_type == 'temporal_qa': # only do temporal eval for temporal_qa
        sub_task_type = 'temporal'
        if pred=="":
            print("no pred")
            score = 0
        else:
            acc, score = check_ans(gt=gt, pred=pred, question=question, sub_task_type=sub_task_type, model=model) # acc is bool, score is given by chatgpt
        # update the scores in result_list for this sample
        res['scores'] = res.get('scores', {})
        res['scores'][sub_task_type] = score
    elif task_type == 'consistency_qa': # only do consistency eval for consistency_qa
        sub_task_type = 'consistency'
        assert 'pred1' in res and 'question1' in res, 'two questions and preds'
        pred1 = res['pred1']
        question1 = res['question1']
        if pred=="" or pred1=="":
            print("no pred")
            score = 0 
        else:
            acc, score = check_ans(
                gt=gt, pred=pred, pred1=pred1, question=question, question1=question1,
                sub_task_type=sub_task_type, model=model) # acc is bool, score is given by chatgpt
        # update the scores in result_list for this sample
        res['scores'] = res.get('scores', {})
        res['scores'][sub_task_type] = score
    else:
        raise NotImplementedError(f'not implemented task type for {task_type}')

    return res

def save_results(result_list, save_path, model="gpt-3.5-turbo-0125"):
    dump_json(result_list, save_path, 'inference_results.json')
    with Pool(7) as pool:
        # result_list = pool.map(partial(chatgpt_eval, model=model), result_list)
        func = partial(chatgpt_eval, model=model)
        result_list = [ res for res in tqdm.tqdm(pool.imap_unordered(func, result_list), total=len(result_list), desc='Language Chat Model Automated Evaluation...')]

    final_res, acc_dict = {}, {}
    correct, total, total_score = 0, 0, 0
    for i, res in enumerate(result_list):
        task_type = res['task_type']
        for sub_task_type, score in res['scores'].items():
            if sub_task_type not in acc_dict:
                acc_dict[sub_task_type] = {
                    'correct': 0,
                    'total': 0,
                    'score': 0,
                } # correct, total
            correct = score > 3
            acc_dict[sub_task_type]['total'] += 1
            acc_dict[sub_task_type]['correct'] += correct
            acc_dict[sub_task_type]['score'] += score
     
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
    result_post =f"-{model}"
    dump_json(all_results, save_path, f'final_results{result_post}.json')
    dump_json(final_res, save_path, f'upload_leaderboard{result_post}.json')

def load_results(save_path, model="gpt-3.5-turbo-0125"):

    result_list = load_json(save_path, f'final_results-{model}.json')
    if result_list is not None:
        result_list = result_list['result_list']
    
    if result_list is None:
        result_list = load_json(save_path, 'inference_results.json')   

    return result_list

class VideoChatGPTBenchDataset(EvalDataset):
    data_dir = "DATAS/VCGBench"
    data_list_info = OrderedDict({
        "generic_qa": OrderedDict(
            json_relpath="Zero_Shot_QA/Benchmarking_QA/generic_qa.json", 
            prefix="DATAS/VCGBench/Videos/Benchmarking", 
            data_type="video", 
            bound=False,
            question_key='Q',
            answer_key='A',
            name_key='video_name',
            postfix=('mp4', 'mkv'),
        ),
        "temporal_qa": OrderedDict(
            json_relpath="Zero_Shot_QA/Benchmarking_QA/temporal_qa.json", 
            prefix="DATAS/VCGBench/Videos/Benchmarking", 
            data_type="video", 
            bound=False,
            question_key='Q',
            answer_key='A',
            name_key='video_name',
            postfix=('mp4', 'mkv'),
        ), # don't has start & end
        "consistency_qa":  OrderedDict( 
            # consistency is quite different in evaluating, and also awkward, hold to later.
            json_relpath="Zero_Shot_QA/Benchmarking_QA/consistency_qa.json", 
            prefix="DATAS/VCGBench/Videos/Benchmarking", 
            data_type="video", 
            bound=False,
            question_key=('Q1', 'Q2'),
            answer_key='A',
            name_key='video_name',
            postfix=('mp4', 'mkv'),
        ),
    })

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        data_list_info = self.data_list_info
        data_dir = self.data_dir

        self.data_list = []
        for k, v in data_list_info.items():
            with open(os.path.join(data_dir, v['json_relpath']), 'r') as f:
                json_data = json.load(f)
            for data in json_data:
                self.data_list.append({
                    'task_type': k,
                    'data': data,
                    **v, # all the infos
                })
        # self.data_list = self.data_list[:10] # for debug
        # random.shuffle(self.data_list) # for debug
        self.decord_method = {
            'video': self.read_video,
            'gif': self.read_gif,
            'frame': self.read_frame,
        }
        # # transform
        # crop_size = resolution
        # scale_size = resolution
        # input_mean = [0.48145466, 0.4578275, 0.40821073]
        # input_std = [0.26862954, 0.26130258, 0.27577711]
        # self.transform = T.Compose([
        #     GroupScale(int(scale_size), interpolation=InterpolationMode.BICUBIC),
        #     GroupCenterCrop(crop_size),
        #     Stack(),
        #     ToTorchFormatTensor(),
        #     GroupNormalize(input_mean, input_std) 
        # ])
    
    def __getitem__(self, idx):
        task_type = self.data_list[idx]['task_type']
        video_name_key = self.data_list[idx]['name_key']
        video_name = self.data_list[idx]['data'][video_name_key]
        video_postfixs = self.data_list[idx]['postfix']
        
        if self.num_segments != 0:
            video_paths =  []
            for p in video_postfixs:
                video_path = os.path.join(self.data_list[idx]['prefix'], video_name + '.' + p)
                if os.path.exists(video_path):
                    video_paths.append(video_path) 
            assert len(video_paths) > 0, f'no video named {video_name}'
            # video_filename = self.data_list[idx]['data'][video_name_key] + video_postfix
            video_path = video_paths[0]
            decord_method = self.decord_method[self.data_list[idx]['data_type']]
            bound = None
            if self.data_list[idx]['bound']:
                bound = (
                    self.data_list[idx]['data']['start'],
                    self.data_list[idx]['data']['end'],
                )
            images_group = decord_method(video_path, bound)
        else:
            # zero frame, no image
            images_group = None

        data = {
            'video_path': video_path,
            'video_pils': images_group, # some might use the original pils and do their own transforms
            'task_type': task_type,
        }


        answer_key = self.data_list[idx]['answer_key']
        question_key = self.data_list[idx]['question_key']
        
        if task_type == 'consistency_qa' and isinstance(question_key, tuple):
            question=self.data_list[idx]['data'][question_key[0]]
            question1=self.data_list[idx]['data'][question_key[1]]
            answer=self.data_list[idx]['data'][answer_key]    

            data.update({
                'question': question, 
                'question1': question1, 
                'answer': answer,
            })
        elif isinstance(question_key, str):
            question=self.data_list[idx]['data'][question_key]
            answer=self.data_list[idx]['data'][answer_key]
            data.update({
                'question': question, 
                'answer': answer,
            })
        else:
            raise ValueError('')

        return data
