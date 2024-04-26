# Data
## Instruction Training Data
> *originated from [Videochat2](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat2)*
>
> [![Dataset meta](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-VideoChat2%20IT-blue)](https://huggingface.co/datasets/OpenGVLab/VideoChat2-IT) 
>
> ![images](./assert/data.png)


We leveraged the training data from [Videochat2](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat2). We only used the video part for video instruct tuning.

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