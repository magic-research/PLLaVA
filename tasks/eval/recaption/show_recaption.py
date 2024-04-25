
import argparse
import gradio as gr

from tasks.eval.recaption import load_results
import json

# example = videogallery().example_inputs()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--save_path',
        required=True,
    )
    args = parser.parse_args()
    return args


args = parse_args()
result_list = load_results(args.save_path)


def show(result_index, ):
    info = result_list[result_index]
    video_path = info['video_path']
    info_str = json.dumps(info, indent=4)
    return video_path, info_str



from tasks.eval.recaption import load_results

with gr.Blocks() as demo:
    gr.Markdown("# Showing of what has came out.")
    gr.Markdown(f"From Saved Results {args.save_path}")
    with gr.Row():
        with gr.Column(1):
            show_video = gr.Video(interactive=False)

        with gr.Column():
            result_index = gr.Slider(0, len(result_list), step=1)
            info = gr.Text(interactive=False)
        
        result_index.change(show, [result_index], [show_video, info])





demo.launch(share=True)
