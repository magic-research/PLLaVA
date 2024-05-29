import json
import os
import os.path as osp
import gc
import logging
import torch
from transformers.integrations import is_deepspeed_zero3_enabled
logger = logging.getLogger(__name__)

@torch.no_grad
def load_state_dict_into_model(model_to_load, state_dict, start_prefix):
    # copied and altered from:
    #   https://github.com/huggingface/transformers/blob/9d35edbb30625489bf286a9b15aed0c5a3119c1c/src/transformers/modeling_utils.py#L650
    #   https://github.com/baaivision/EVA/blob/2ca37a8c0d82b9496754f3fa9c3966b4caa54d75/EVA-CLIP-18B/shinji/eva_clip/factory.py#L168

    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata
    error_msgs = []
    # PyTorch's `_load_from_state_dict` does not copy parameters in a module's descendants
    # so we need to apply the function recursively.
    def load(module: torch.nn.Module, prefix=""):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        args = (state_dict, prefix, local_metadata, True, [], [], error_msgs)
        # Parameters of module and children will start with prefix. We can exit early if there are none in this state_dict
        if is_deepspeed_zero3_enabled():
            import deepspeed
            with deepspeed.zero.GatheredParameters(list(module.parameters(recurse=False)), modifier_rank=0):
                if torch.distributed.get_rank() == 0:
                    module._load_from_state_dict(*args)
        else:
            module._load_from_state_dict(*args)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + ".")

    load(model_to_load, prefix=start_prefix)
    # Delete `state_dict` so it could be collected by GC earlier. Note that `state_dict` is a copy of the argument, so
    # it's safe to delete it.
    del state_dict
    return error_msgs

@torch.no_grad
def load_from_pretrained(model, folder, prefer_safe=True):
    """
    COPIED and ALTERED FROM https://github.com/huggingface/transformers/blob/bdb9106f247fca48a71eb384be25dbbd29b065a8/src/transformers/modeling_utils.py#L417
    This is the same as
    [`torch.nn.Module.load_state_dict`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=load_state_dict#torch.nn.Module.load_state_dict)
    but for a sharded checkpoint.

    This load is performed efficiently: each checkpoint shard is loaded one by one in RAM and deleted after being
    loaded in the model.

    Args:
        model (`torch.nn.Module`): The model in which to load the checkpoint.
        folder (`str` or `ospLike`): A path to a folder containing the sharded checkpoint.
        strict (`bool`, *optional`, defaults to `True`):
            Whether to strictly enforce that the keys in the model state dict match the keys in the sharded checkpoint.
        prefer_safe (`bool`, *optional*, defaults to `False`)
            If both safetensors and PyTorch save files are present in checkpoint and `prefer_safe` is True, the
            safetensors files will be loaded. Otherwise, PyTorch files are always loaded when possible.

    Returns:
        `NamedTuple`: A named tuple with `missing_keys` and `unexpected_keys` fields
            - `missing_keys` is a list of str containing the missing keys
            - `unexpected_keys` is a list of str containing the unexpected keys
    """

    logger.info(f"loading pretrained weights from pretrained path {folder}")

    # Load the index
    from safetensors.torch import load_file as safe_load_file
    from transformers.utils import (
        WEIGHTS_INDEX_NAME,
        SAFE_WEIGHTS_INDEX_NAME,
        is_safetensors_available, 
    )
    from transformers.pytorch_utils import is_torch_greater_or_equal_than_1_13
    from functools import partial
    index_file = osp.join(folder, WEIGHTS_INDEX_NAME)
    safe_index_file = osp.join(folder, SAFE_WEIGHTS_INDEX_NAME)

    index_present = osp.isfile(index_file)
    safe_index_present = osp.isfile(safe_index_file)
    
    if not index_present and not (safe_index_present and is_safetensors_available()):
        filenames = (
            (WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_INDEX_NAME) if is_safetensors_available() else (WEIGHTS_INDEX_NAME,)
        )
        logger.warning(f"Can't find a checkpoint index ({' or '.join(filenames)}) in {folder}. Assume not sharded, directly loading from model.safetensors. Be aware: also not checking for key matching.")

        shard_files = ['model.safetensors']
        strict, load_safe = False, True
    else:
        strict, load_safe = True, False # strict if whole model is saved (sharded)
        if safe_index_present:
            if prefer_safe:
                if is_safetensors_available():
                    load_safe = True  # load safe due to preference
                else:
                    logger.warning(
                        f"Cannot load sharded checkpoint at {folder} safely since safetensors is not installed!"
                    )
            elif not index_present:
                load_safe = True  # load safe since we have no other choice

        load_index = safe_index_file if load_safe else index_file

        with open(load_index, "r", encoding="utf-8") as f:
            index = json.load(f)

        shard_files = list(set(index["weight_map"].values()))

        # If strict=True, error before loading any of the state dicts.
        loaded_keys = index["weight_map"].keys()
        model_keys = model.state_dict().keys()
        missing_keys = [key for key in model_keys if key not in loaded_keys]
        unexpected_keys = [key for key in loaded_keys if key not in model_keys]
        if strict and (len(missing_keys) > 0 or len(unexpected_keys) > 0):
            error_message = f"Error(s) in loading state_dict for {model.__class__.__name__}"
            if len(missing_keys) > 0:
                str_missing_keys = ",".join([f'"{k}"' for k in missing_keys])
                error_message += f"\nMissing key(s): {str_missing_keys}."
            if len(unexpected_keys) > 0:
                str_unexpected_keys = ",".join([f'"{k}"' for k in unexpected_keys])
                error_message += f"\nUnexpected key(s): {str_unexpected_keys}."
            raise RuntimeError(error_message)

    loader = (
        safe_load_file
        if load_safe
        else partial(torch.load, map_location="cpu", weights_only=is_torch_greater_or_equal_than_1_13)
    )

    for shard_file in shard_files:
        shard_filepath = osp.join(folder, shard_file)
        state_dict = loader(shard_filepath)
        # SOMETIMES RUN THROUGH, SOMETIMES DOESN'T WHY???
        # load_zero_partitions(model, state_dict, True, 'test')
        error_message = load_state_dict_into_model(model, state_dict, '')
        if len(error_message) != 0:
            raise RuntimeError(f"Error Loading matched key parameters in {shard_filepath}: {error_message}")
        else:
            logger.info(f"Successfully Loaded from {shard_filepath}")
        # Make sure memory is freed before we load the next state dict.
        del state_dict
        gc.collect()
