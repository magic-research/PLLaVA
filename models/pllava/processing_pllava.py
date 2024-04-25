# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Processor class for Llava.
"""


import itertools
from typing import List, Optional, Union
import PIL.Image
import numpy as np

from transformers import AutoTokenizer
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import (
    ImageInput,
    make_list_of_images,
    valid_images,
    infer_channel_dimension_format,
    to_numpy_array,
    get_image_size,
    ChannelDimension,
)
from transformers.image_processing_utils import get_size_dict
from transformers.image_utils import PILImageResampling
from transformers.processing_utils import ProcessorMixin
from transformers.image_transforms import resize, pad, PaddingMode, to_channel_dimension_format, get_resize_output_image_size
from transformers.tokenization_utils_base import PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from transformers.utils import TensorType


class PllavaProcessor(ProcessorMixin):
    r"""
    Constructs a Llava processor which wraps a Llava image processor and a Llava tokenizer into a single processor.

    [`LlavaProcessor`] offers all the functionalities of [`CLIPImageProcessor`] and [`LlamaTokenizerFast`]. See the
    [`~LlavaProcessor.__call__`] and [`~LlavaProcessor.decode`] for more information.

    Args:
        image_processor ([`CLIPImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`LlamaTokenizerFast`], *optional*):
            The tokenizer is a required input.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "CLIPImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor=None, tokenizer=None, 
                 shortest_edge=336,
                 longest_edge=762,
                 center_pad=False):
        self.shortest_edge = shortest_edge
        self.longest_edge = longest_edge
        self.center_pad = center_pad
        super().__init__(image_processor, tokenizer)

    def resize_crop_longshort(self, videos: list[list[np.ndarray]], input_data_format):
        video_spatial_sizes = [get_image_size(images[0], input_data_format) for images in videos]
        long_short_rates = [max(size) / min(size) for size in video_spatial_sizes]
        min_long_short_rate = min(long_short_rates)
        min_long_short_video_idx = long_short_rates.index(min_long_short_rate)

        clip_resolution = self.image_processor.size['shortest_edge']
        out_video_spatial_size = video_spatial_sizes[min_long_short_video_idx]
        out_videos_short_edge = max(min(size) for size in video_spatial_sizes)
        resize_longest_edge = max(max(size) for size in video_spatial_sizes)
        resize_longest_edge = min(640, resize_longest_edge)
        out_videos_short_edge = min(out_videos_short_edge, int(resize_longest_edge / min_long_short_rate))
        out_videos_short_edge = max(out_videos_short_edge, clip_resolution)

    
        if out_video_spatial_size[0] > out_video_spatial_size[1]: # h > w:
            out_video_spatial_size = (int(out_videos_short_edge * min_long_short_rate), out_videos_short_edge )
        else:
            out_video_spatial_size = ( out_videos_short_edge, int(out_videos_short_edge * min_long_short_rate) )
        videos = [
            [self.resize(frame, input_data_format=input_data_format, shortest_edge=out_videos_short_edge, longest_edge=9999) for frame in frames]
            for frames in videos
        ]
        out_videos = []
        for frames in videos:
            out_frames = []
            video_spatial_size = get_image_size(frames[0], input_data_format)
            assert min(video_spatial_size) == out_videos_short_edge
            overhead = (max(video_spatial_size) - max(out_video_spatial_size)) // 2
            slice_start, slice_end = overhead // 2,   overhead // 2 + max(out_video_spatial_size)
            hslice, wslice = (slice(slice_start, slice_end), slice(None, None)) if video_spatial_size[0] > video_spatial_size[1] \
                             else (slice(None, None), slice(slice_start, slice_end)) # h > w
            for frame in frames:
                if input_data_format == ChannelDimension.FIRST:
                    out_frames.append(frame[..., hslice, wslice])
                elif input_data_format == ChannelDimension.LAST:
                    out_frames.append(frame[..., hslice, wslice, :])
            out_videos.append(out_frames)

        return out_videos

    @staticmethod
    def _compute_num_blocks_and_overlaps(input_shape, resolution):
        input_shape = np.array(input_shape)
        resolution = np.array(resolution)
        assert input_shape.max() >= resolution
        num_blocks = np.ceil(input_shape / resolution).astype(np.int32).tolist()
        overlaps = [0 if size % resolution==0 
                    else int(np.floor((resolution - size % resolution) / (num_block - 1))) for num_block, size in zip(num_blocks, input_shape)]
        return num_blocks, overlaps

    def resize(
        self,
        image: np.ndarray,
        resample: PILImageResampling = PILImageResampling.BICUBIC, # type: ignore
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        shortest_edge: int = None,
        longest_edge: int = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Resize an image. The shortest edge of the image is resized to size["shortest_edge"], with the longest edge
        resized to keep the input aspect ratio.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Size of the output image.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
                Resampling filter to use when resiizing the image.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        shortest_edge = getattr(self, 'shortest_edge', None) if shortest_edge is None else shortest_edge
        longest_edge = getattr(self, 'longest_edge', None) if longest_edge is None else longest_edge
        default_to_square = False
        output_size = get_resize_output_image_size(
            image,
            size=shortest_edge,
            default_to_square=default_to_square,
            max_size=longest_edge,
            input_data_format=input_data_format,
        )
        clip_resolution = self.image_processor.size['shortest_edge']
        if min(output_size) < clip_resolution:
            output_size = get_resize_output_image_size(
                image,
                size=shortest_edge,
                default_to_square=default_to_square,
                input_data_format=input_data_format,
            )
        return resize(
            image,
            size=output_size,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        images: ImageInput = None,
        center_pad = None,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length=None,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to LlamaTokenizerFast's [`~LlamaTokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the image(s), this method forwards the `images` and `kwrags` arguments to
        CLIPImageProcessor's [`~CLIPImageProcessor.__call__`] if `images` is not `None`. Please refer to the doctsring
        of the above two methods for more information.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. In case of a NumPy array/PyTorch tensor, each image should be of shape (C, H, W), where C is a
                number of channels, H and W are image height and width.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `False`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding
                index) among:
                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length (see above).
            truncation (`bool`, *optional*):
                Activates truncation to cut input sequences longer than `max_length` to `max_length`.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
        """
        data=dict()
        if images is not None:
            if isinstance(images, list) and isinstance(images[0], PIL.Image.Image):
                videos = [images] # one video
            else:
                videos = images
            
            pixel_values_list = []
            videos = [[to_numpy_array(image) for image in make_list_of_images(images)] for images in videos]
            # images = [self.resize(image, ) if min(get_image_size(image, input_data_format)) < clip_resolution else image for image in images]
            input_data_format = infer_channel_dimension_format(videos[0][0])
            videos = self.resize_crop_longshort(videos, input_data_format)

            for images in videos:
                if not valid_images(images):
                    raise ValueError(
                        "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                        "torch.Tensor, tf.Tensor or jax.ndarray."
                    )                

                center_pad = center_pad if center_pad is not None else self.center_pad
                if center_pad:
                    images = [self.pad_to_square(image, 0, input_data_format, input_data_format) for image in images]

                pixel_values = self.image_processor(images, return_tensors='np')["pixel_values"]
                pixel_values_list.append(pixel_values)

            pixel_values = np.concatenate(pixel_values_list)
            data.update(pixel_values=pixel_values)
            
        else:
            data.update(pixel_values = None)

        if text is not None:
            text_inputs = self.tokenizer(
                text, return_tensors=return_tensors, padding=padding, truncation=truncation, max_length=max_length
            )
            data.update(**text_inputs)
        return BatchFeature(data, tensor_type=return_tensors)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.batch_decode with CLIP->Llama
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.decode with CLIP->Llama
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.model_input_names
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
