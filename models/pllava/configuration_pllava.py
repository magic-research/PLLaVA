# coding=utf-8
# Copyright 2023 Microsoft Research & University of Wisconsin-Madison and the HuggingFace Inc. team. All rights reserved.
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
""" Llava model configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers.models.auto import CONFIG_MAPPING


logger = logging.get_logger(__name__)

PLLAVA_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "llava-hf/llava-v1.5-7b": "https://huggingface.co/llava-hf/llava-v1.5-7b/resolve/main/config.json",
}


class PllavaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`LlavaForConditionalGeneration`]. It is used to instantiate an
    Llava model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Llava-9B.

    e.g. [llava-hf/llava-9b](https://huggingface.co/llava-hf/llava-9b)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vision_config (`LlavaVisionConfig`,  *optional*):
            Custom vision config or dict
        text_config (`Union[AutoConfig, dict]`, *optional*):
            The config object of the text backbone. Can be any of `LlamaConfig` or `MistralConfig`.
        ignore_index (`int`, *optional*, defaults to -100):
            The ignore index for the loss function.
        image_token_index (`int`, *optional*, defaults to 32000):
            The image token index to encode the image prompt.
        projector_hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The activation function used by the multimodal projector.
        vision_feature_select_strategy (`str`, *optional*, defaults to `"default"`):
            The feature selection strategy used to select the vision feature from the CLIP backbone.
        vision_feature_layer (`int`, *optional*, defaults to -2):
            The index of the layer to select the vision feature.
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the Llava model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`~LlavaForConditionalGeneration`]

    Example:

    ```python
    >>> from transformers import LlavaForConditionalGeneration, LlavaConfig, CLIPVisionConfig, LlamaConfig

    >>> # Initializing a CLIP-vision config
    >>> vision_config = CLIPVisionConfig()

    >>> # Initializing a Llama config
    >>> text_config = LlamaConfig()

    >>> # Initializing a Llava llava-1.5-7b style configuration
    >>> configuration = LlavaConfig(vision_config, text_config)

    >>> # Initializing a model from the llava-1.5-7b style configuration
    >>> model = LlavaForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "pllava"
    is_composition = False

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        ignore_index=-100,
        image_token_index=32000,
        projector_hidden_act="gelu",
        vision_feature_select_strategy="default",
        vision_feature_layer=-2,
        vocab_size=32000,
        pooling_method='avg',
        pooling_shape=(8, 16, 16),
        frame_shape=(24, 24), # llava 1.5 pretrained frame shape
        num_frames=1, # llava 1.5 pretrained frame shape
        use_pooling=True,
        gradient_checkpointing=False,
        **kwargs,
    ):
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.projector_hidden_act = projector_hidden_act
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.vision_feature_layer = vision_feature_layer
        self.vocab_size = vocab_size
        self.use_pooling = use_pooling
        self.gradient_checkpointing = gradient_checkpointing
        
        self.vision_config = vision_config
        
        self.pooling_method = pooling_method # should be in 'max', 'avg'
        self.pooling_shape = pooling_shape # 
        self.frame_shape = frame_shape # 
        self.num_frames = num_frames
        if isinstance(self.vision_config, dict):
            vision_config["model_type"] = (
                vision_config["model_type"] if "model_type" in vision_config else "clip_vision_model"
            )
            self.vision_config = CONFIG_MAPPING[vision_config["model_type"]](**vision_config)
        elif vision_config is None:
            self.vision_config = CONFIG_MAPPING["clip_vision_model"](
                intermediate_size=4096,
                hidden_size=1024,
                patch_size=14,
                image_size=336,
                num_hidden_layers=24,
                num_attention_heads=16,
                vocab_size=32000,
                projection_dim=768,
            )
        self.vocab_size = self.vocab_size

        self.text_config = text_config
        
        if isinstance(self.text_config, dict):
            text_config["model_type"] = text_config["model_type"] if "model_type" in text_config else "llama"
            self.text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
            self.vocab_size = self.text_config.vocab_size
            self.text_config.gradient_checkpointing = self.gradient_checkpointing
            
        elif text_config is None:
            tmp_config = {"_attn_implementation":"flash_attention_2",
                          "gradient_checkpointing": self.gradient_checkpointing}
            self.text_config = CONFIG_MAPPING["llama"](**tmp_config)
            self.text_config.gradient_checkpointing = self.gradient_checkpointing
        # self.text_config["_attn_implementation"]="flash_attention_2"  # xl: temporal hard code
        

        super().__init__(**kwargs)
