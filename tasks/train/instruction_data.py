import os as __os  # add "__" if not want to be exported
from copy import deepcopy as __deepcopy
import itertools as __itertools

data_root = "DATAS/TRAIN_TEST"
anno_root_it = f"{data_root}/magic_jsons"

# ============== pretraining datasets=================
available_corpus = dict(
    # image
    # caption_coco=[
    #     f"{anno_root_it}/image/caption/coco/train.json", 
    #     f"{data_root}/images/coco",
    # ],
    # caption_llava=[
    #     f"{anno_root_it}/image/caption/llava/train.json", 
    #     f"{data_root}/images/coco",
    # ],
    # caption_minigpt4=[
    #     f"{anno_root_it}/image/caption/minigpt4/train.json", 
    #     f"{data_root}/images/minigpt4_align/image",
    # ],
    # caption_paragraph_captioning=[
    #     f"{anno_root_it}/image/caption/paragraph_captioning/train.json", 
    #     f"{data_root}/images/m3it/image-paragraph-captioning",
    # ],
    # caption_textcaps=[
    #     f"{anno_root_it}/image/caption/textcaps/train.json", 
    #     f"{data_root}/images/textcaps",
    # ],
    # classification_imagenet=[
    #     f"{anno_root_it}/image/classification/imagenet/train.json", 
    #     f"{data_root}/images/m3it/imagenet",
    # ],
    # classification_coco_itm=[
    #     f"{anno_root_it}/image/classification/coco_itm/train.json", 
    #     f"{data_root}/images/coco",
    # ],
    # conversation_llava=[
    #     f"{anno_root_it}/image/conversation/llava/train.json", 
    #     f"{data_root}/images/coco",
    # ],
    # reasoning_clevr=[
    #     f"{anno_root_it}/image/reasoning/clevr/train.json", 
    #     f"{data_root}/images/m3it/clevr",
    # ],
    # reasoning_visual_mrc=[
    #     f"{anno_root_it}/image/reasoning/visual_mrc/train.json", 
    #     f"{data_root}/images/m3it/visual_mrc",
    # ],
    # reasoning_llava=[
    #     f"{anno_root_it}/image/reasoning/llava/train.json", 
    #     f"{data_root}/images/coco",
    # ],
    # vqa_vqav2=[
    #     f"{anno_root_it}/image/vqa/vqav2/train.json", 
    #     f"{data_root}/images/m3it/vqav2",
    # ],
    # vqa_gqa=[
    #     f"{anno_root_it}/image/vqa/gqa/train.json", 
    #     f"{data_root}/images/gqa/images",
    # ],
    # vqa_okvqa=[
    #     f"{anno_root_it}/image/vqa/okvqa/train.json", 
    #     f"{data_root}/images/m3it/okvqa",
    # ],
    # vqa_a_okvqa=[
    #     f"{anno_root_it}/image/vqa/a_okvqa/train.json", 
    #     f"{data_root}/images/m3it/a_okvqa",
    # ],
    # vqa_viquae=[
    #     f"{anno_root_it}/image/vqa/viquae/train.json", 
    #     f"{data_root}/images/viquae_images",
    # ],
    # vqa_ocr_vqa=[
    #     f"{anno_root_it}/image/vqa/ocr_vqa/train.json", 
    #     f"{data_root}/images/ocr_vqa/images",
    # ],
    # vqa_text_vqa=[
    #     f"{anno_root_it}/image/vqa/text_vqa/train.json", 
    #     f"{data_root}/images/textvqa",
    # ],
    # vqa_st_vqa=[
    #     f"{anno_root_it}/image/vqa/st_vqa/train.json", 
    #     f"{data_root}/images/m3it/st-vqa",
    # ],
    # vqa_docvqa=[
    #     f"{anno_root_it}/image/vqa/docvqa/train.json", 
    #     f"{data_root}/images/docvqa",
    # ],
    # origin_llava=[
    #     f"{anno_root_it}/image/origin_llava/train.json", 
    #     f"{data_root}/images",
    # ],
    # video
    caption_textvr=[
        f"{anno_root_it}/video/caption/textvr/train.json", 
        f"{data_root}/videos/TextVR",
        "video"
    ],
    caption_videochat=[
        f"{anno_root_it}/video/caption/videochat/train.json", 
        f"{data_root}/videos/webvid_10m",
        "video"
    ], # not ready, need to read from hdfs
    caption_webvid=[
        f"{anno_root_it}/video/caption/webvid/train.json", 
        f"{data_root}/videos/webvid_10m",
        "video"
    ], # not ready, need to read from hdfs
    caption_youcook2=[
        f"{anno_root_it}/video/caption/youcook2/train.json", 
        f"{data_root}/videos/YouCook2/split_videos",
        "video"
    ],
    classification_k710=[
        f"{anno_root_it}/video/classification/k710/train.json", 
        f"{data_root}/videos/kinetics",
        "video"
    ],
    classification_ssv2=[
        f"{anno_root_it}/video/classification/ssv2/train.json", 
        f"{data_root}/videos/20bn-something-something-v2",
        "video"
    ],
    conversation_videochat1=[
        f"{anno_root_it}/video/conversation/videochat1/train.json", 
        f"{data_root}/videos/webvid_10m",
        "video"
    ],# not ready, need to read from hdfs
    conversation_videochat2=[
        f"{anno_root_it}/video/conversation/videochat2/train.json", 
        f"{data_root}/videos/InternVid-10M-FLT/videos",
        "video"
    ],
    conversation_videochatgpt=[
        f"{anno_root_it}/video/conversation/videochatgpt/train.json", 
        f"{data_root}/videos/AVideo_ChatGPT",
        "video"
    ],
    reasoning_next_qa=[
        f"{anno_root_it}/video/reasoning/next_qa/train.json", 
        f"{data_root}/videos/NExTVideo",
        "video"
    ],
    reasoning_clevrer_qa=[
        f"{anno_root_it}/video/reasoning/clevrer_qa/train.json", 
        f"{data_root}/videos/CLEVRER",
        "video"
    ],
    reasoning_clevrer_mc=[
        f"{anno_root_it}/video/reasoning/clevrer_mc/train.json",  
        f"{data_root}/videos/CLEVRER",
        "video"
    ],
    vqa_ego_qa=[
        f"{anno_root_it}/video/vqa/ego_qa/train.json", 
        f"{data_root}/videos/ego4d_data/split_videos",
        "video"
    ],
    vqa_tgif_frame_qa=[
        f"{anno_root_it}/video/vqa/tgif_frame_qa/train.json", 
        f"{data_root}/videos/tgif",
        "video"
    ],
    vqa_tgif_transition_qa=[
        f"{anno_root_it}/video/vqa/tgif_transition_qa/train.json", 
        f"{data_root}/videos/tgif",
        "video"
    ],
    vqa_webvid_qa=[
        f"{anno_root_it}/video/vqa/webvid_qa/train.json", 
        f"{data_root}/videos/webvid_10m",
        "video"
    ],# not ready, need to read from hdfs
    origin_videochatgpt=[
        f"{anno_root_it}/video/origin_videochatgpt/train.json", 
        f"{data_root}/videos/Video_ChatGPT",
        "video"
    ],
)



available_corpus["videochat2_instruction_full"] = [
    available_corpus["caption_coco"],
    available_corpus["caption_llava"],
    available_corpus["caption_minigpt4"],
    available_corpus["caption_paragraph_captioning"],
    available_corpus["caption_textcaps"],
    available_corpus["classification_imagenet"],
    available_corpus["classification_coco_itm"],
    available_corpus["conversation_llava"],
    available_corpus["reasoning_clevr"],
    available_corpus["reasoning_visual_mrc"],
    available_corpus["reasoning_llava"],
    available_corpus["vqa_vqav2"],
    available_corpus["vqa_gqa"],
    available_corpus["vqa_okvqa"],
    available_corpus["vqa_a_okvqa"],
    available_corpus["vqa_viquae"],
    available_corpus["vqa_ocr_vqa"],
    available_corpus["vqa_text_vqa"],
    available_corpus["vqa_st_vqa"],
    available_corpus["vqa_docvqa"],
    available_corpus["caption_textvr"],
    available_corpus["caption_youcook2"],
    available_corpus["classification_k710"],
    available_corpus["classification_ssv2"],
    available_corpus["conversation_videochat2"],
    available_corpus["conversation_videochatgpt"],
    available_corpus["reasoning_next_qa"],
    available_corpus["reasoning_clevrer_qa"],
    available_corpus["reasoning_clevrer_mc"],
    available_corpus["vqa_ego_qa"],
    available_corpus["vqa_tgif_frame_qa"],
    available_corpus["vqa_tgif_transition_qa"],
    available_corpus["conversation_videochat1"],
    available_corpus["vqa_webvid_qa"],
    available_corpus["caption_videochat"],
    available_corpus["caption_webvid"],
]

available_corpus["videochat2_video"] = [
    available_corpus["caption_textvr"],
    available_corpus["caption_youcook2"],
    available_corpus["classification_k710"],
    available_corpus["classification_ssv2"],
    available_corpus["conversation_videochat2"],
    available_corpus["conversation_videochatgpt"],
    available_corpus["reasoning_next_qa"],
    available_corpus["reasoning_clevrer_qa"],
    available_corpus["reasoning_clevrer_mc"],
    available_corpus["vqa_ego_qa"],
    available_corpus["vqa_tgif_frame_qa"],
    available_corpus["vqa_tgif_transition_qa"],
    available_corpus["conversation_videochat1"],
    available_corpus["vqa_webvid_qa"],
    available_corpus["caption_videochat"],
    available_corpus["caption_webvid"],
]




# ============== for debug=================
available_corpus["videochat2_instruction_debug"] = [
    # available_corpus["caption_minigpt4"],
    available_corpus["caption_textvr"],
    # available_corpus["vqa_ego_qa"],
    # available_corpus["classification_k710"],
    # available_corpus["reasoning_next_qa"],
    # available_corpus["caption_textvr"],
    # available_corpus["caption_youcook2"],

    # available_corpus["caption_textcaps"], # realistic caption foucsing in real life text
    # available_corpus["caption_textvr"], # good realistic captioning, also focusing on text
]


if __name__ == '__main__':
    print(len(list(
        __itertools.chain(
        available_corpus['conversation_data'],
        available_corpus['reasoning_data'],
        available_corpus['conversation_videochat2'],
        available_corpus['caption_data'],
        available_corpus['classification_data'],
    )
    )))
    print(len(available_corpus['videochat2_instruction_full']))