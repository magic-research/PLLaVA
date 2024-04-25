

import argparse
import json
import os
import os.path as osp
import gradio as gr

from tasks.eval.recaption import load_results as load_results_recaption
from tasks.eval.mvbench import load_results as load_results_mvbench
from tasks.eval.vcgbench import load_results as load_results_vcgbench
from tasks.eval.videoqabench import load_results as load_results_videoqabench

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

def show(result_list, result_index):
    info = result_list[result_index]
    video_path = info['video_path']
    info_str = json.dumps(info, indent=4)
    return video_path, info_str

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
            break

    result_index = gr.Slider(0, len(result_list), step=1)

    return result_list, result_index

with gr.Blocks() as demo:
    result_list = gr.State()

    with gr.Row():
        gr.Markdown("# Showing of what has came out.")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(f"### From Saved Results Directory {args.root_dir}")
          
        with gr.Column(scale=2):
            result_dir = gr.Dropdown(label='Test Result Path')
            button_reload = gr.Button(value='Reload From The Evaluation/Inference Root Directory')



    with gr.Row():
        with gr.Column():
            show_video = gr.Video(interactive=False)

        with gr.Column():
            result_index = gr.Slider(0, 0, step=1, label="Index")
            info = gr.Text(interactive=False, label='Detailed Output Information')
        

    button_reload.click(reload_results_dirs, [], [result_dir])
    result_dir.change(reload_results, [result_dir], [result_list, result_index])
    result_index.change(show, [result_list, result_index], [show_video, info])
    demo.load(reload_results_dirs, [], [result_dir])
    
demo.launch(share=True)