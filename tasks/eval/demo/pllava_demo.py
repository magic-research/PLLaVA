from argparse import ArgumentParser
import copy
import gradio as gr
from gradio.themes.utils import colors, fonts, sizes

from utils.easydict import EasyDict
from tasks.eval.model_utils import load_pllava
from tasks.eval.eval_utils import (
    ChatPllava,
    conv_plain_v1,
    Conversation,
    conv_templates
)
from tasks.eval.demo import pllava_theme

SYSTEM="""You are Pllava, a large vision-language assistant. 
You are able to understand the video content that the user provides, and assist the user with a variety of tasks using natural language.
Follow the instructions carefully and explain your answers in detail based on the provided video.
"""
INIT_CONVERSATION: Conversation = conv_plain_v1.copy()


# ========================================
#             Model Initialization
# ========================================
def init_model(args):

    print('Initializing PLLaVA')
    model, processor = load_pllava(
        args.pretrained_model_name_or_path, args.num_frames, 
        use_lora=args.use_lora, 
        weight_dir=args.weight_dir, 
        lora_alpha=args.lora_alpha, 
        use_multi_gpus=args.use_multi_gpus)
    if not args.use_multi_gpus:
        model = model.to('cuda')
    chat = ChatPllava(model, processor)
    return chat


# ========================================
#             Gradio Setting
# ========================================
def gradio_reset(chat_state, img_list):
    if chat_state is not None:
        chat_state = INIT_CONVERSATION.copy()
    if img_list is not None:
        img_list = []
    return (
        None,
        gr.update(value=None, interactive=True),
        gr.update(value=None, interactive=True),
        gr.update(placeholder='Please upload your video first', interactive=False),
        gr.update(value="Upload & Start Chat", interactive=True),
        chat_state,
        img_list
    )


def upload_img(gr_img, gr_video, chat_state=None, num_segments=None, img_list=None):
    print(gr_img, gr_video)
    chat_state = INIT_CONVERSATION.copy() if chat_state is None else chat_state
    img_list = [] if img_list is None else img_list
    
    if gr_img is None and gr_video is None:
        return None, None, gr.update(interactive=True),gr.update(interactive=True, placeholder='Please upload video/image first!'), chat_state, None
    if gr_video: 
        llm_message, img_list, chat_state = chat.upload_video(gr_video, chat_state, img_list, num_segments)
        return (
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True, placeholder='Type and press Enter'),
            gr.update(value="Start Chatting", interactive=False),
            chat_state,
            img_list,
        )
    if gr_img:
        llm_message, img_list,chat_state = chat.upload_img(gr_img, chat_state, img_list)
        return (
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True, placeholder='Type and press Enter'),
            gr.update(value="Start Chatting", interactive=False),
            chat_state,
            img_list
        )


def gradio_ask(user_message, chatbot, chat_state, system):
    if len(user_message) == 0:
        return gr.update(interactive=True, placeholder='Input should not be empty!'), chatbot, chat_state
    chat_state =  chat.ask(user_message, chat_state, system)
    chatbot = chatbot + [[user_message, None]]
    return '', chatbot, chat_state


def gradio_answer(chatbot, chat_state, img_list, num_beams, temperature):
    llm_message, llm_message_token, chat_state = chat.answer(conv=chat_state, img_list=img_list, max_new_tokens=200, num_beams=num_beams, temperature=temperature)
    llm_message = llm_message.replace("<s>", "") # handle <s>
    chatbot[-1][1] = llm_message
    print(chat_state)
    print(f"Answer: {llm_message}")
    return chatbot, chat_state, img_list


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        default='llava-hf/llava-1.5-7b-hf'
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
        "--use_multi_gpus",
        action='store_true'
    )
    parser.add_argument(
        "--weight_dir",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--conv_mode",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--server_port",
        type=int,
        required=False,
        default=7868,
    )
    args = parser.parse_args()
    return args


title = """<h1 align="center"><a href="https://github.com/magic-research/PLLaVA"><img src="https://raw.githubusercontent.com/magic-research/PLLaVA/main/assert/logo.png" alt="PLLAVA" border="0" style="margin: 0 auto; height: 100px;" /></a> </h1>"""
description = (
    """<br><p><a href='https://github.com/magic-research/PLLaVA'>
    # PLLAVA!
    <img src='https://img.shields.io/badge/Github-Code-blue'></a></p><p>
    - Upload A Video
    - Press Upload
    - Start Chatting
    """
)

args = parse_args()

model_description = f"""
    # MODEL INFO
    - pretrained_model_name_or_path:{args.pretrained_model_name_or_path}
    - use_lora:{args.use_lora}
    - weight_dir:{args.weight_dir}
"""

# with gr.Blocks(title="InternVideo-VideoChat!",theme=gvlabtheme,css="#chatbot {overflow:auto; height:500px;} #InputVideo {overflow:visible; height:320px;} footer {visibility: none}") as demo:
with gr.Blocks(title="PLLaVA",
               theme=pllava_theme,
               css="#chatbot {overflow:auto; height:500px;} #InputVideo {overflow:visible; height:320px;} footer {visibility: none}") as demo:
    gr.Markdown(title)
    gr.Markdown(description)
    gr.Markdown(model_description)
    with gr.Row():
        with gr.Column(scale=0.5, visible=True) as video_upload:
            # with gr.Column(elem_id="image", scale=0.5) as img_part:
            with gr.Tab("Video", elem_id='video_tab'):
                up_video = gr.Video(interactive=True, include_audio=True, elem_id="video_upload", height=360)
            with gr.Tab("Image", elem_id='image_tab'):
                up_image = gr.Image(type="pil", interactive=True, elem_id="image_upload", height=360)
            upload_button = gr.Button(value="Upload & Start Chat", interactive=True, variant="primary")
            clear = gr.Button("Restart")
            
            # num_segments = gr.Slider(
            #     minimum=8,
            #     maximum=64,
            #     value=8,
            #     step=1,
            #     interactive=True,
            #     label="Video Segments",
            # )
        
        with gr.Column(visible=True)  as input_raws:
            system_string = gr.Textbox(SYSTEM, interactive=True, label='system')
            num_beams = gr.Slider(
                minimum=1,
                maximum=5,
                value=1,
                step=1,
                interactive=True,
                label="beam search numbers",
            )           
            temperature = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=1.0,
                step=0.1,
                interactive=True,
                label="Temperature",
            )
            
            chat_state = gr.State()
            img_list = gr.State()
            chatbot = gr.Chatbot(elem_id="chatbot",label='Conversation')
            with gr.Row():
                with gr.Column(scale=0.7):
                    text_input = gr.Textbox(show_label=False, placeholder='Please upload your video first', interactive=False, container=False)
                with gr.Column(scale=0.15, min_width=0):
                    run = gr.Button("💭Send")
                with gr.Column(scale=0.15, min_width=0):
                    clear = gr.Button("🔄Clear")     
    
    with gr.Row():
        examples = gr.Examples(
            examples=[
                ['example/jesse_dance.mp4', 'What is the man doing?'],
                ['example/yoga.mp4', 'What is the woman doing?'],
                ['example/cooking.mp4', 'Describe the background, characters and the actions in the provided video.'],
                # ['example/cooking.mp4', 'What is happening in the video?'],
                ['example/working.mp4', 'Describe the background, characters and the actions in the provided video.'],
                ['example/1917.mov', 'Describe the background, characters and the actions in the provided video.'],
            ],
            inputs=[up_video, text_input]
        )


    chat = init_model(args)
    INIT_CONVERSATION = conv_templates[args.conv_mode]
    upload_button.click(upload_img, [up_image, up_video, chat_state], [up_image, up_video, text_input, upload_button, chat_state, img_list])
    
    text_input.submit(gradio_ask, [text_input, chatbot, chat_state, system_string], [text_input, chatbot, chat_state]).then(
        gradio_answer, [chatbot, chat_state, img_list, num_beams, temperature], [chatbot, chat_state, img_list]
    )
    run.click(gradio_ask, [text_input, chatbot, chat_state, system_string], [text_input, chatbot, chat_state]).then(
        gradio_answer, [chatbot, chat_state, img_list, num_beams, temperature], [chatbot, chat_state, img_list]
    )
    run.click(lambda: "", None, text_input)  
    clear.click(gradio_reset, [chat_state, img_list], [chatbot, up_image, up_video, text_input, upload_button, chat_state, img_list], queue=False)

# demo.queue(max_size=5)
demo.launch(share=True,server_port=args.server_port)
# demo.launch(server_name="0.0.0.0", server_port=10034, enable_queue=True)
