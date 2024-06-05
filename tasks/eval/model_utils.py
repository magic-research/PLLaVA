
import torch
import os
from safetensors import safe_open
from accelerate import init_empty_weights, dispatch_model, infer_auto_device_map,load_checkpoint_in_model
from accelerate.utils import get_balanced_memory
from transformers import StoppingCriteria
from transformers.modeling_utils import load_sharded_checkpoint
from peft import get_peft_model, LoraConfig, TaskType
from peft import PeftModel

from tasks.model_utils import load_from_pretrained
from tasks.eval.eval_utils import Conversation
from models.pllava import PllavaProcessor, PllavaForConditionalGeneration, PllavaConfig





class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.tokenizer = tokenizer
        self.start_len = None
        self.input_ids = input_ids

    def __call__(
        self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        if self.start_len is None:
            self.start_len = self.input_ids.shape[1]
            return False
        else:
            outputs = self.tokenizer.batch_decode(output_ids[:, self.start_len:], skip_special_tokens=False)
            flag = True
            for output in outputs:
                for keyword in self.keywords:
                    if keyword not in output:
                        flag = False
                        return False
            return flag


def load_pllava(
        repo_id,
        num_frames,
        use_lora=False,
        lora_dir=None,
        weight_dir=None,
        lora_alpha=32,
        use_multi_gpus=False,
        pooling_shape=(16,12,12)
    ):
    kwargs = {
        'num_frames': num_frames,
    }
    # print("===============>pooling_shape", pooling_shape)
    if num_frames == 0:
        kwargs.update(pooling_shape=(0,12,12)) # produce a bug if ever usen the pooling projector
    config = PllavaConfig.from_pretrained(
        repo_id,
        pooling_shape=pooling_shape,
        **kwargs,
    )
    
    with torch.no_grad():
        model = PllavaForConditionalGeneration.from_pretrained(repo_id, config=config, torch_dtype=torch.bfloat16)
        processor = PllavaProcessor.from_pretrained(repo_id)

    # config lora
    if use_lora:
        print("Use lora")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False,  target_modules=["q_proj", "v_proj"],
            r=128, lora_alpha=lora_alpha, lora_dropout=0.
        )
        print("Lora Scaling:", lora_alpha/128)
        if lora_dir is not None:
            model.language_model = PeftModel.from_pretrained(model.language_model, lora_dir, config=peft_config)
        else:
            model.language_model = get_peft_model(model.language_model, peft_config)
        print("Finish constructing lora")
    
    # load weights
    if weight_dir is not None:
        print(f'loading checkpoint from {weight_dir}')
        load_from_pretrained(model, weight_dir, strict=not use_lora)
        print(f'done loading')

    # dispatch model weight
    if use_multi_gpus:
        max_memory = get_balanced_memory(
            model,
            max_memory=None,
            no_split_module_classes=["LlamaDecoderLayer"],
            dtype='bfloat16',
            low_zero=False,
        )
        for k, mem in max_memory.items():
            if isinstance(k, int):
                max_memory[k] = mem - 2_000_000_000 # leave out 2G for inference
                
        device_map = infer_auto_device_map(
            model,
            max_memory=max_memory,
            no_split_module_classes=["LlamaDecoderLayer"],
            dtype='bfloat16'
        )

        dispatch_model(
            model,
            device_map=device_map,
            offload_dir="tmp/offload"
        )
        print(model.hf_device_map)

    model = model.eval()

    return model, processor


def load_adapters(model, adapter_model_name_or_paths):

    for adapter_model_name_or_path in adapter_model_name_or_paths:
        if not isinstance(model, PeftModel):
            model = PeftModel.from_pretrained(model, adapter_model_name_or_path, adapter_model_name_or_path)
        else:
            model.load_adapter(adapter_model_name_or_path, adapter_model_name_or_path)

    return model


def pllava_answer(conv: Conversation, model, processor, img_list, do_sample=True, max_new_tokens=200, num_beams=1, min_length=1, top_p=0.9,
               repetition_penalty=1.0, length_penalty=1, temperature=1.0, stop_criteria_keywords=None, print_res=False):
    # torch.cuda.empty_cache()
    prompt = conv.get_prompt()
    inputs = processor(text=prompt, images=img_list, return_tensors="pt")
    if inputs['pixel_values'] is None:
        inputs.pop('pixel_values')
    inputs = inputs.to(device=model.device, dtype=model.dtype)
    generation_kwargs = {
        "do_sample": do_sample,
        "max_new_tokens": max_new_tokens,
        "num_beams": num_beams,
        "min_length": min_length,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "length_penalty": length_penalty,
        "temperature": temperature,
        "eos_token_id": processor.tokenizer.convert_tokens_to_ids(conv.sep[1]),
        "stopping_criteria": [KeywordsStoppingCriteria(stop_criteria_keywords + conv.sep, processor.tokenizer, inputs.input_ids)] if stop_criteria_keywords is not None \
                              else [ KeywordsStoppingCriteria(conv.sep, processor.tokenizer, inputs.input_ids)] 

    }

    with torch.no_grad():
        output_token = model.generate(**inputs, media_type='video', **generation_kwargs, )
        output_text = processor.batch_decode(output_token, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]

    if print_res: # debug usage
        print('### PROMPTING LM WITH: ', prompt)
        print('### LM OUTPUT TEXT:  ', output_text)
    if conv.roles[-1] == "<|im_start|>assistant\n":
        split_tag = "<|im_start|> assistant\n"
    else:
        split_tag = conv.roles[-1]
    output_text = output_text.split(split_tag)[-1]
    ending = conv.sep if isinstance(conv.sep, str) else conv.sep[1]
    output_text = output_text.removesuffix(ending).strip()
    conv.messages[-1][1] = output_text
    return output_text, conv

