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



### Recaptioning
#### Inter4k

This is a dataset with 1000 samples of high resolution videos. We prepare the data folloing the instructions from their [official website](https://alexandrosstergiou.github.io/datasets/Inter4K/index.html)

#### Extending Reacptioning
The recaptioning part is designed to be extendable.

inference script [tasks/eval/recaption/pllava_recaption.py](tasks/eval/recaption/pllava_recaption.py) would use a dataset class [RecaptionDataset](tasks/eval/recaption/__init__.py#L197). The detailed information is kept in the data_list_info attribute as:
```
data_list_info = OrderedDict({
        # "Panda70M": OrderedDict(
        #     json_relpath="Panda70M/annotations.json", 
        #     prefix="DATAS/Recaption/Panda70M/videos", 
        #     data_type="video", 
        #     bound=False,
        #     key_rename_map={
        #         # 'caption': 'hint',
        #     },
        #     name_key='video_name',
        #     postfix=('mp4', 'mkv', 'webm'),
        #     recaption_type=RecaptionSample,
        # ), # don't has start & end
        "Inter4K": OrderedDict(
            json_relpath="Inter4K/annotations.json", 
            prefix="DATAS/Recaption/Inter4K/60fps/UHD", 
            data_type="video", 
            bound=False,
            key_rename_map={
                # 'caption': 'hint',
            },
            name_key='video_name',
            postfix=('mp4', 'mkv', 'webm'),
            recaption_type=CaptionSample,
        ), # don't has start & end
    })
```
It contains the path to a annotation json file where there is a list and each item of the list is a sample waiting for captioning. For example, the Inter4K/annotations.json is like:
```json
[
    {
        "video_name": "973"
    },
    ...
]
```
and the directory DATAS/Recaption/Inter4K/60fps/UHD would look like:
```
$ ls DATAS/Recaption/Inter4K/60fps/UHD
1.mp4 134.mp4  170.mp4 ....
```

Naively, only the video is needed when captioning directly, therefore the annotation file only needs to contain the names of each video under the "prefix" directory.

Extending a dataset for captioning would consist of the folloing steps:
1. have all the videos downloaded
2. construct a annotation.json file with sepecific format.
3. configure the recaption dataset [here](tasks/eval/recaption/__init__.py#L197), where you would need to determine:
    - json_relpath: the annotation relative path
    - prefix: root directory for videos
    - postfix: a list containing all the file extensions for these videos

The other options are experimental, so stick with the default setting as in Inter4k. The recommended length of video is around 5-20 seconds. 

p.s. "bound" is to make sure the video pass to the model doesn't have scene transition or so. This part wasn't tested, so set the bound to false and make sure the original videos files are single clip of a video. But always feel free to discover and contribute to PLLaVA!