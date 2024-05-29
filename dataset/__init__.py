import torch
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from dataset.it_dataset import ITImgTrainDataset, ITVidTrainDataset


def get_media_type(dataset_config):
    if len(dataset_config) == 3 and dataset_config[2] == "video":
        return "video"
    elif dataset_config[-1] == "only_video":
        return "only_video"
    else:
        return "image"


def create_dataset(dataset_type, config):
    if "clip" in config.model.get("vit_model", 'vit'):
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        vision_enc_name = config.model.vision_encoder.name
        if "swin" in vision_enc_name or "vit" in vision_enc_name:
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
        elif "beit" in vision_enc_name:
            mean = (0.5, 0.5, 0.5)  # for all beit model except IN1K finetuning
            std = (0.5, 0.5, 0.5)
        elif "clip" in vision_enc_name:
            mean = (0.48145466, 0.4578275, 0.40821073)
            std = (0.26862954, 0.26130258, 0.27577711)
        else:
            raise ValueError

    normalize = transforms.Normalize(mean, std)

    # loaded images and videos are torch.Tensor of torch.uint8 format,
    # ordered as (T, 1 or 3, H, W) where T=1 for image
    type_transform = transforms.Lambda(lambda x: x.float().div(255.0))

    if config.inputs.video_input.random_aug:
        aug_transform = transforms.RandAugment()
    else:
        aug_transform = transforms.Lambda(lambda x: x)

    train_transform = transforms.Compose(
        [
            aug_transform,
            transforms.RandomResizedCrop(
                config.inputs.image_res,
                scale=(0.5, 1.0),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            type_transform,
            normalize,
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize(
                (config.inputs.image_res, config.inputs.image_res),
                interpolation=InterpolationMode.BICUBIC,
            ),
            type_transform,
            normalize,
        ]
    )

    video_reader_type = config.inputs.video_input.get("video_reader_type", "decord")
    video_only_dataset_kwargs_train = dict(
        video_reader_type=video_reader_type,
        sample_type=config.inputs.video_input.sample_type,
        num_frames=config.inputs.video_input.num_frames,
        num_tries=3,  # false tolerance
    )

    if dataset_type == "pt_train":
        raise ValueError("NOT PRETRAINING YET")
    elif dataset_type in ["it_train"]:
        # convert to list of lists
        train_files = (
            [config.train_file] if isinstance(config.train_file[0], str) else config.train_file
        )
        train_media_types = sorted(list({get_media_type(e) for e in train_files}))

        train_datasets = []
        for m in train_media_types:
            dataset_cls = ITImgTrainDataset if m == "image" else ITVidTrainDataset
            # dataset of the same media_type will be mixed in a single Dataset object
            _train_files = [e for e in train_files if get_media_type(e) == m]

            datasets = []
            for train_file in _train_files:
                dataset_kwargs = dict(
                    ann_file=train_file,
                    transform=train_transform,
                    mm_alone=config.preprocess.get("mm_alone", True),
                    add_second_msg=config.preprocess.get("add_second_msg", True),
                    skip_short_sample=config.preprocess.get("skip_short_sample", False),
                    clip_transform=config.preprocess.get("clip_transform", False),
                    random_shuffle=config.preprocess.get("random_shuffle", True),
                    system=config.preprocess.get("system", ""),
                    role=config.preprocess.get('roles', ("Human", "Assistant")),
                    end_signal=config.preprocess.get('end_signal', "###"),
                    begin_signal=config.preprocess.get('begin_signal', ""),
                )
                if m == "video":
                    video_only_dataset_kwargs_train.update({
                        "start_token": config.model.get("start_token", "<Video>"),
                        "end_token": config.model.get("end_token", "</Video>"),
                    })
                    dataset_kwargs.update(video_only_dataset_kwargs_train)
                    if "tgif" in train_file[1]:
                        video_only_dataset_kwargs_train.update({
                            "video_reader_type": "gif"
                        })
                        dataset_kwargs.update(video_only_dataset_kwargs_train)
                    elif False: # elif "webvid" in train_file[1]:
                        video_only_dataset_kwargs_train.update({
                            "video_reader_type": "hdfs"
                        })
                    else:
                        video_only_dataset_kwargs_train.update({
                            "video_reader_type": "decord"
                        })
                    dataset_kwargs.update(video_only_dataset_kwargs_train)
                datasets.append(dataset_cls(**dataset_kwargs))
            dataset = ConcatDataset(datasets)
            train_datasets.append(dataset)
        return train_datasets


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(
        datasets, samplers, batch_size, num_workers, is_trains, collate_fns
    ):
        if is_train:
            shuffle = sampler is None
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=False,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
            persistent_workers=True if n_worker > 0 else False,
        )
        loaders.append(loader)
    return loaders

