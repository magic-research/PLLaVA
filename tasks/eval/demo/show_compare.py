

import argparse
import json
import os
import os.path as osp
import gradio as gr
import numpy as np

from tasks.eval.recaption import load_results as load_results_recaption
from tasks.eval.mvbench import load_results as load_results_mvbench
from tasks.eval.vcgbench import load_results as load_results_vcgbench
from tasks.eval.videoqabench import load_results as load_results_videoqabench
from tasks.eval.demo import pllava_theme


load_results_funcs = [
    load_results_recaption,
    load_results_mvbench,
    load_results_vcgbench,
    load_results_videoqabench,
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root_dir',
        required=True,
    )
    args = parser.parse_args()
    return args

args = parse_args()
root_dir = args.root_dir

def show(result_list_first, result_list_second, result_index):
    sample2index_second = {}

    for i, result in enumerate(result_list_second):
        if 'video_path' not in result:
            continue

        question = result['question'] if 'question' in result else ''
        video_path = result['video_path']
        samplehash = question + '--' +video_path
        sample2index_second[samplehash] = i

    info = result_list_first[result_index]
    info_str_first = json.dumps(info, indent=4, ensure_ascii=False)
    video_path = info['video_path']
    question = info['question'] if 'question' in info else ''
    samplehash = question + '--' +video_path
    if samplehash in sample2index_second:
        info = result_list_second[sample2index_second[samplehash]]
        info_str_second = json.dumps(info, indent=4, ensure_ascii=False)
    else:
        info_str_second = f"NO {video_path} IN THE SECOND RESULT DIR"
    return video_path, info_str_first, info_str_second

def reload_results_dirs():
    result_dirs = []
    # load result dir paths
    for dirpath, dirnames, filenames in os.walk(args.root_dir):
        if len(dirnames) == 0 and len(filenames) != 0:
            result_dirs.append(dirpath)
    return gr.Dropdown(result_dirs, value=result_dirs[0])

def reload_results(result_dir):
    # if isinstance(result_dir, list):
    #     result_dir = result_dir[0]

    if result_dir is None or not osp.exists(result_dir):
        return None
    
    for fn in load_results_funcs:
        result_list = fn(result_dir)
        if result_list is not None:
            np.random.shuffle(result_list)
            break
    result_index = gr.Slider(0, len(result_list), step=1)

    return result_list, result_index



with gr.Blocks(title="PLLAVA RESULTS", theme=pllava_theme) as demo:
    result_list_first = gr.State()
    result_list_second = gr.State()

    with gr.Row():
        with gr.Column():
            gr.Markdown("# Showing off Model's Outputs.")
            gr.Markdown(
                "You can find all our results, including:\n"
                "1. results of Captioned Inter4k\n"
                "2. results of Different Benchmark inference outputs.\n"
                "Choose a directory to see the different output variant.\n"
                "You can also choose secondary directory (as long as they are from the same dataset.) to compare on the results.\n"
            )

    with gr.Row():
        with gr.Column():
            show_video = gr.Video(interactive=False)

        with gr.Column():
            button_reload = gr.Button(value='Reload From The Evaluation/Inference Root Directory')
            result_index = gr.Slider(0, 0, step=1, label="Index")

            result_dir_first = gr.Dropdown(label='Test Result Path')
            info_first = gr.Text(interactive=False, label='Detailed Output Information')
            result_dir_second = gr.Dropdown(label='Test Result Path')
            info_second = gr.Text(interactive=False, label='Detailed Output Information')
        

    button_reload.click(reload_results_dirs, [], [result_dir_first])
    button_reload.click(reload_results_dirs, [], [result_dir_second])
    result_dir_first.change(reload_results, [result_dir_first], [result_list_first, result_index])
    result_dir_second.change(reload_results, [result_dir_second], [result_list_second, result_index])
    result_index.change(show, [result_list_first, result_list_second, result_index], [show_video, info_first, info_second])
    demo.load(reload_results_dirs, [], [result_dir_first])
    demo.load(reload_results_dirs, [], [result_dir_second])
    
demo.launch(share=True)