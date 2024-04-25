# Data
## Instruction Training Data
> *originated from [Videochat2](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat2)*
>
> [![Dataset meta](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-VideoChat2%20IT-blue)](https://huggingface.co/datasets/OpenGVLab/VideoChat2-IT) 
>
> ![images](./assert/data.png)


----

### Annotations
A comprehensive dataset of **1.9M** data annotations is available in [JSON](https://huggingface.co/datasets/OpenGVLab/VideoChat2-IT) format. Due to the extensive size of the full data, we provide only JSON files [here](https://huggingface.co/datasets/OpenGVLab/VideoChat2-IT). For corresponding images and videos, please follow our instructions.

### Source Video
:warning: **For the used videos of InternVid, EgoQA and YouCook2, please check [issues](https://github.com/OpenGVLab/Ask-Anything/issues/86#issuecomment-1882529070).**

We treated video datasets differently. Please download the original videos from the provided links:
- [VideoChat](https://github.com/OpenGVLab/InternVideo/tree/main/Data/instruction_data): Based on [InternVid](https://github.com/OpenGVLab/InternVideo/tree/main/Data/InternVid), we created additional instruction data and used GPT-4 to condense the existing data.
- [VideoChatGPT](https://github.com/mbzuai-oryx/Video-ChatGPT/tree/main/data): The original caption data was converted into conversation data based on the same VideoIDs.
- [Kinetics-710](https://github.com/OpenGVLab/UniFormerV2/blob/main/DATASET.md) & [SthSthV2](https://developer.qualcomm.com/software/ai-datasets/something-something): Option candidates were generated from [UMT](https://github.com/OpenGVLab/unmasked_teacher) top-20 predictions.
- [NExTQA](https://github.com/doc-doc/NExT-QA): Typos in the original sentences were corrected.
- [CLEVRER](https://clevrer.csail.mit.edu/): For single-option multiple-choice QAs, we used only those concerning color/material/shape. For multi-option multiple-choice QAs, we utilized all the data.
- [WebVid](https://maxbain.com/webvid-dataset/): Non-overlapping data was selected for captioning and [QA](https://antoyang.github.io/just-ask.html#webvidvqa).
- [YouCook2](https://youcook2.eecs.umich.edu/): Original videos were truncated based on the official dense captions.
- [TextVR](https://github.com/callsys/textvr): All data was used without modifications.
- [TGIF](https://github.com/YunseokJANG/tgif-qa): Only TGIF$_{frame}$ and TGIF$_{Transition}$ subsets were considered.
- [EgoQA](https://ego4d-data.org/): Some egocentric QAs were generated from Ego4D data.

For all datasets, task instructions were automatically generated using GPT-4.

## Evaluation Data & Others
Follow this section to obtain the evaluation open resources.

### VCGBench

We refer to the VideoChatGPT video question answering evaluation as VCGBench in this repo. We followed the original [repo](https://github.com/mbzuai-oryx/Video-ChatGPT/tree/main) to prepare the evaluation data.

### MVBench
We follow the original [Videochat2 repo](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat2) in setting up the MVBench Evaluation. You can also find helpful resources at their [huggingface repo](https://huggingface.co/datasets/OpenGVLab/MVBench)


### Videoqabench
We refer to all other video question answering benchmarks as videoqabench in this repo. They are mainly prepared folloing the original repos. Each listed:
1. [MSVD](https://www.cs.utexas.edu/users/ml/clamp/videoDescription/) & [MSRVTT](https://github.com/xudejing/video-question-answering)

3. [Activity Net](https://github.com/MILVLG/activitynet-qa/tree/master)
4. [TGIF](https://github.com/raingo/TGIF-Release/tree/master)

Also other fantastic repo intergrating these benchmarks are helpful in the process of setting up the evaluation data:
- [VideoChatGPT](https://github.com/mbzuai-oryx/Video-ChatGPT/tree/main)
- [VideoLlava](https://github.com/PKU-YuanGroup/Video-LLaVA/tree/main/videollava)
- [IG-VLM](https://github.com/imagegridworth/IG-VLM/tree/main)

### Inter4k

This is a dataset with 1000 samples of high resolution videos. We prepare the data folloing the instructions from their [official website](https://alexandrosstergiou.github.io/datasets/Inter4K/index.html)