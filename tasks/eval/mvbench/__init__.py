import os
import json
from tasks.eval.eval_utils import (
    dump_json,
    load_json,
    EvalDataset,
)


def check_ans(pred, gt):
    flag = False
    
    pred_list = pred.lower().split(' ')
    pred_option, pred_content = pred_list[0], ' '.join(pred_list[1:])
    gt_list = gt.lower().split(' ')
    gt_option, gt_content = gt_list[0], ' '.join(gt_list[1:])
    if gt_content[-1] == '.':
        gt_content = gt_content[:-1]
    
    if not any([c in pred_option for c in 'abcdefgABCDEFG']):
        print(f"model doesn't follow instructions: {pred}")
    elif pred_option.replace('.', '') in gt_option:
        flag = True
    elif gt_option in pred_option:
        flag = True
        
    return flag

def save_results(result_list, save_path):

    final_res, acc_dict = {}, {}
    correct, total = 0, 0
    for res in result_list:
        task_type = res['task_type']
        if task_type not in acc_dict:
            acc_dict[task_type] = [0, 0] # correct, total
        acc_dict[task_type][1] += 1
        total += 1
        pred = res['pred']
        gt = res['gt']
        if check_ans(pred=pred, gt=gt):
            acc_dict[task_type][0] += 1
            correct += 1

    for k, v in acc_dict.items():
        final_res[k] = v[0] / v[1] * 100
        correct += v[0]
        total += v[1]    
    final_res['Avg'] = correct / total * 100

    all_results = {
        "acc_dict": acc_dict,
        "result_list": result_list
    }
    dump_json(all_results, save_path, 'all_results.json')
    dump_json(final_res, save_path, 'upload_leaderboard.json')

def load_results(save_path):
    all_results = load_json(save_path, 'all_results.json')
    if all_results is not None:
        result_list = all_results['result_list']
    else:
        result_list = None
    # json_data = load_json(save_path, 'all_results.json')['result_list']
    return result_list

class MVBenchDataset(EvalDataset):
    data_list_info = {
        # "task_type (sub task name)": ("json file name", "image/video prefix", "data_type", "bound")
        "Action Sequence": ("action_sequence.json", "DATAS/MVBench/video/star/Charades_v1_480/", "video", True), # has start & end
        "Action Prediction": ("action_prediction.json", "DATAS/MVBench/video/star/Charades_v1_480/", "video", True), # has start & end
        "Action Antonym": ("action_antonym.json", "DATAS/MVBench/video/ssv2_video/", "video", False),
        "Fine-grained Action": ("fine_grained_action.json", "DATAS/MVBench/video/Moments_in_Time_Raw/videos/", "video", False),
        "Unexpected Action": ("unexpected_action.json", "DATAS/MVBench/video/FunQA_test/test/", "video", False),
        "Object Existence": ("object_existence.json", "DATAS/MVBench/video/clevrer/video_validation/", "video", False),
        "Object Interaction": ("object_interaction.json", "DATAS/MVBench/video/star/Charades_v1_480/", "video", True), # has start & end
        "Object Shuffle": ("object_shuffle.json", "DATAS/MVBench/video/perception/videos/", "video", False),
        "Moving Direction": ("moving_direction.json", "DATAS/MVBench/video/clevrer/video_validation/", "video", False),
        "Action Localization": ("action_localization.json", "DATAS/MVBench/video/sta/sta_video/", "video", True),  # has start & end
        "Scene Transition": ("scene_transition.json", "DATAS/MVBench/video/scene_qa/video/", "video", False),
        "Action Count": ("action_count.json", "DATAS/MVBench/video/perception/videos/", "video", False),
        "Moving Count": ("moving_count.json", "DATAS/MVBench/video/clevrer/video_validation/", "video", False),
        "Moving Attribute": ("moving_attribute.json", "DATAS/MVBench/video/clevrer/video_validation/", "video", False),
        "State Change": ("state_change.json", "DATAS/MVBench/video/perception/videos/", "video", False),
        "Fine-grained Pose": ("fine_grained_pose.json", "DATAS/MVBench/video/nturgbd/", "video", False),
        "Character Order": ("character_order.json", "DATAS/MVBench/video/perception/videos/", "video", False),
        "Egocentric Navigation": ("egocentric_navigation.json", "DATAS/MVBench/video/vlnqa/", "video", False),
        "Episodic Reasoning": ("episodic_reasoning.json", "DATAS/MVBench/video/tvqa/frames_fps3_hq/", "frame", True),  # has start & end, read frame
        "Counterfactual Inference": ("counterfactual_inference.json", "DATAS/MVBench/video/clevrer/video_validation/", "video", False),
    }
    data_dir = "DATAS/MVBench/json"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        data_list_info = self.data_list_info
        data_dir = self.data_dir

        self.data_list = []
        for k, v in data_list_info.items():
            with open(os.path.join(data_dir, v[0]), 'r') as f:
                json_data = json.load(f)
            for data in json_data:
                self.data_list.append({
                    'task_type': k,
                    'prefix': v[1],
                    'data_type': v[2],
                    'bound': v[3],
                    'data': data
                })
        # self.data_list = self.data_list[:100] # for debug
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
        question, answer = self.qa_template(self.data_list[idx]['data'])
        task_type = self.data_list[idx]['task_type']
        decord_method = self.decord_method[self.data_list[idx]['data_type']]
        bound = None
        if self.data_list[idx]['bound']:
            bound = (
                self.data_list[idx]['data']['start'],
                self.data_list[idx]['data']['end'],
            )
        video_path = os.path.join(self.data_list[idx]['prefix'], self.data_list[idx]['data']['video'])


        # images_group = decord_method(video_path, bound)
        try: # might be problem with decord
            images_group = decord_method(video_path, bound)
        except Exception as e:
            print(f'error decoding {video_path}')
            task_type = 'error_reading_video'
            images_group = None

        return {
            'video_path': video_path, 
            'video_pils': images_group, # some might use the original pils and do their own transforms
            'question': question, 
            'answer': answer,
            'task_type': task_type,
        }
        

    def qa_template(self, data):
        question = f"Question: {data['question']}\n"
        question += "Options:\n"
        answer = data['answer']
        answer_idx = -1
        for idx, c in enumerate(data['candidates']):
            question += f"({chr(ord('A') + idx)}) {c}\n"
            if c == answer:
                answer_idx = idx
        question = question.rstrip()
        answer = f"({chr(ord('A') + answer_idx)}) {answer}"
        return question, answer

