import logging
import os
import json
import random

import numpy as np

from dataset.base_dataset import ImageVideoBaseDataset
from dataset.video_utils import VIDEO_READER_FUNCS

logger = logging.getLogger(__name__)
IMAGE_TOKEN="<image>"

class ITImgTrainDataset(ImageVideoBaseDataset):
    media_type = "image"

    def __init__(
        self,
        ann_file,
        transform, 
        system="", role=("Human", "Assistant"),
        mm_alone=True,
        add_second_msg=True,
        start_token="<Image>", end_token="</Image>",
        random_shuffle=True, # if True, shuffle the QA list ##xl:????? why need random shuffle
        begin_signal=None,
        end_signal=None,
        clip_transform=False,
        skip_short_sample=False,
    ):
        super().__init__()
        self.mm_alone = mm_alone
        self.clip_transform = clip_transform
        if len(ann_file) == 3 and ann_file[2] == "video":
            self.media_type = "video"  
        else:
            self.media_type = "image"
        self.label_file, self.data_root = ann_file[:2]

        logger.info(f'Load annotations from json file  {self.label_file}')
        with open(self.label_file, 'r') as f:
            self.anno = json.load(f)
        self.num_examples = len(self.anno)
        self.transform = transform
        annos = []
        logger.info(f'Validating existing data in {self.label_file}')
        for i, ann in enumerate(self.anno):
            filename = ann['video'] if 'video' in ann else ann['image']
            if False: # if self.media_type =='video' and "webvid" in self.data_root:
                video_id, extension = os.path.splitext(os.path.basename(filename))
                if video_id not in self.keys_indexfile:
                    pass
                else:
                    annos.append(ann)
            else:

                if filename is None or filename=="None":
                    logger.warn(f"caught filename is None in annotation file at index {i} in {self.label_file}")
                else:
                    annos.append(ann)
                    # hack this since remote file systems cost too much time
                    # if os.path.exists(os.path.join(self.data_root, filename)):
                    #     annos.append(ann)
                    # else:
                    #     logger.warn(f"no {os.path.join(self.data_root, filename)}, skipping sample; [In label_file {self.label_file}]")
        self.anno = annos
        self.num_examples = len(self.anno)


        # prompt parameters
        if system:
            assert system[-1] == " ", "' ' should be add in the end of system, thus '###' will be tokenized into one token."
        # currently not support add start_token and end_token in the system, since the msg should be added properly
        self.begin_signal = [begin_signal for _ in role] if isinstance(begin_signal, str) else begin_signal
        self.end_signal = [end_signal for _ in role] if isinstance(end_signal, str) else end_signal
        self.start_token = start_token
        self.end_token = end_token
        self.system = system
        self.role = role
        self.random_shuffle = random_shuffle
        # instruction location and number
        logger.info(f"Random shuffle: {self.random_shuffle}")

    def get_anno(self, index):
        filename = self.anno[index][self.media_type]
        qa = self.anno[index]["QA"]
        
        if "start" in self.anno[index] and "end" in self.anno[index]:
            anno = {
                "image": os.path.join(self.data_root, filename), "qa": qa,
                "start": self.anno[index]["start"], "end": self.anno[index]["end"],
            }
        else:
            anno = {"image": os.path.join(self.data_root, filename), "qa": qa}
        return anno

    def __len__(self):
        return self.num_examples
    
    def process_qa(self, qa, msg=""):
        cur_instruction = ""
        # randomly shuffle qa for conversation
        if self.random_shuffle and len(qa) > 1:
            random.shuffle(qa)
        if "i" in qa[0].keys() and qa[0]["i"] != "":
            cur_instruction = qa[0]["i"] + self.end_signal[0]

        conversation = self.system
        # add instruction as system message
        if cur_instruction:
            conversation += cur_instruction

        # rstrip() for the extra " " in msg
        if self.mm_alone:
            conversation += (
                self.begin_signal[0] + self.role[0] + 
                self.start_token + self.end_token + msg.rstrip() + self.end_signal[0]
            )

        for i, sentence in enumerate(qa):
            q = self.start_token + self.end_token+"\n"+ qa[0]["q"] if (not self.mm_alone) and (i == 0) else sentence["q"]
            a = sentence["a"]
            if q != "":
                conversation += (self.begin_signal[0] + self.role[0] + q + self.end_signal[1])
            else:
                # no question, often in caption dataset
                pass
            conversation += (self.begin_signal[0] + self.role[1] + a + self.end_signal[1])
        
        
        if cur_instruction:
            cur_instruction += qa[0]["q"]
        return conversation, cur_instruction.strip()

    def __getitem__(self, index):
        try:
            ann = self.get_anno(index)
            image, index = self.load_and_transform_media_data_image(index, ann["image"], clip_transform=self.clip_transform)
            conversation, instruction = self.process_qa(ann["qa"])
            return image, conversation, instruction, index
        except Exception as e:
            logger.warning(f"Caught exception {e} when loading image {ann['image']}")
            index = np.random.randint(0, len(self))
            return self.__getitem__(index)


class ITVidTrainDataset(ITImgTrainDataset):
    media_type = "video"

    def __init__(
        self, ann_file, transform,
        num_frames=4, video_reader_type="decord", sample_type="rand", num_tries=3,
        mm_alone=True,
        system="", role=("Human", "Assistant"),
        start_token="<Video>", end_token="</Video>",
        add_second_msg=True,
        random_shuffle=True,
        begin_signal=None,
        end_signal=None,
        clip_transform=False,
        skip_short_sample=False,
        
    ):
        # # "id index file for webvid"
        # if "webvid" in ann_file[1]:
        #     with open("/mnt/bn/dq-storage-ckpt/xulin/datasets/videos/webvid_10m/keys_indexfile.json") as f:
        #         self.keys_indexfile = json.load(f) # the correponding index file for each webvid id
        
        super().__init__(
            ann_file,
            transform, 
            system=system,
            role=role,
            mm_alone=mm_alone,
            start_token=start_token,
            end_token=end_token,
            random_shuffle=random_shuffle,
            begin_signal=begin_signal,
            end_signal=end_signal,
            clip_transform=clip_transform,
            skip_short_sample=skip_short_sample,
        )
        self.num_frames = num_frames
        self.video_reader_type = video_reader_type
        self.video_reader = VIDEO_READER_FUNCS[video_reader_type]
        self.sample_type = sample_type
        self.num_tries = num_tries
        self.add_second_msg = add_second_msg

        logger.info(f"Use {video_reader_type} for data in {ann_file}")
        if add_second_msg:
            logger.info(f"Add second message: The video contains X frames sampled at T seconds.")

    def __getitem__(self, index):
        try:
            ann = self.get_anno(index)

            msg = ""
            clip = None
            if "start" in ann and "end" in ann:
                clip = [ann["start"], ann["end"]]
            video, index, sec = self.load_and_transform_media_data_video(
                index,
                ann["image"],
                return_fps=True,
                clip=clip,
                clip_transform=self.clip_transform
            )
            
            if self.add_second_msg:
                # " " should be added in the start and end
                msg = f" The video contains {len(sec)} frames sampled at {', '.join(sec)} seconds. "
            conversation, instruction = self.process_qa(ann["qa"], msg)
            return video, conversation, instruction, index
        except Exception as e:
            logger.warning(f"Caught exception {e} when loading video {ann['image']}")
            index = np.random.randint(0, len(self))
            return self.__getitem__(index)