
import torch
import os
from peft import get_peft_model, LoraConfig, TaskType
from safetensors import safe_open
from peft import PeftModel
from tasks.eval.eval_utils import Conversation
from models.plava import PlavaProcessor, PlavaForConditionalGeneration, PlavaConfig
from accelerate import init_empty_weights, dispatch_model, infer_auto_device_map,load_checkpoint_in_model
from accelerate.utils import get_balanced_memory

from transformers import StoppingCriteria
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
            outputs = self.tokenizer.batch_decode(
                output_ids[:, self.start_len:], skip_special_tokens=True
            )
            flag = True
            for output in outputs:
                for keyword in self.keywords:
                    if keyword not in output:
                        flag = False
                        return False
            return flag


def load_plava(repo_id, num_frames, use_lora=False, weight_dir=None, lora_alpha=32, use_multi_gpus=False, pooling_shape=(16,12,12)):
    kwargs = {
        'num_frames': num_frames,
    }
    # print("===============>pooling_shape", pooling_shape)
    if num_frames == 0:
        kwargs.update(pooling_shape=(0,12,12)) # produce a bug if ever usen the pooling projector
    config = PlavaConfig.from_pretrained(
        repo_id if not use_lora else weight_dir,
        pooling_shape=pooling_shape,
        **kwargs,
    )
    
    with torch.no_grad():
        model = PlavaForConditionalGeneration.from_pretrained(repo_id, config=config, torch_dtype=torch.bfloat16)
        
    try:
        processor = PlavaProcessor.from_pretrained(repo_id)
    except Exception as e:
        processor = PlavaProcessor.from_pretrained('llava-hf/llava-1.5-7b-hf')

    # config lora
    if use_lora and weight_dir is not None:
        print("Use lora")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False,  target_modules=["q_proj", "v_proj"],
            r=128, lora_alpha=lora_alpha, lora_dropout=0.
        )
        print("Lora Scaling:", lora_alpha/128)
        model.language_model = get_peft_model(model.language_model, peft_config)
        assert weight_dir is not None, "pass a folder to your lora weight"
        print("Finish use lora")
    
    # load weights
    if weight_dir is not None:
        state_dict = {}
        save_fnames = os.listdir(weight_dir)
        if "model.safetensors" in save_fnames:
            use_full = False
            for fn in save_fnames:
                if fn.startswith('model-0'):
                    use_full=True        
                    break
        else:
            use_full= True

        if not use_full:
            print("Loading weight from", weight_dir, "model.safetensors")
            with safe_open(f"{weight_dir}/model.safetensors", framework="pt", device="cpu") as f:
                for k in f.keys():
                    state_dict[k] = f.get_tensor(k)
        else:
            print("Loading weight from", weight_dir)
            for fn in save_fnames:
                if fn.startswith('model-0'):
                    with safe_open(f"{weight_dir}/{fn}", framework="pt", device="cpu") as f:
                        for k in f.keys():
                            state_dict[k] = f.get_tensor(k)
            
        if 'model' in state_dict.keys():
            msg = model.load_state_dict(state_dict['model'], strict=False)
        else:
            msg = model.load_state_dict(state_dict, strict=False)
        print(msg)
    # dispatch model weight
    if use_multi_gpus:
        max_memory = get_balanced_memory(
            model,
            max_memory=None,
            no_split_module_classes=["LlamaDecoderLayer"],
            dtype='bfloat16',
            low_zero=False,
        )

        device_map = infer_auto_device_map(
            model,
            max_memory=max_memory,
            no_split_module_classes=["LlamaDecoderLayer"],
            dtype='bfloat16'
        )

        dispatch_model(model, device_map=device_map)
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


def plava_answer(conv: Conversation, model, processor, img_list, do_sample=True, max_new_tokens=200, num_beams=1, min_length=1, top_p=0.9,
               repetition_penalty=1.0, length_penalty=1, temperature=1.0, stop_criteria_keywords=None, print_res=False, fea_save_path=None):
    # torch.cuda.empty_cache()
    prompt = conv.get_prompt()
    inputs = processor(text=prompt, images=img_list, return_tensors="pt")
    if inputs['pixel_values'] is None:
        inputs.pop('pixel_values')
    inputs = inputs.to(model.device)
    
    # set up stopping criteria
    if stop_criteria_keywords is not None:
        stopping_criteria = [KeywordsStoppingCriteria(stop_criteria_keywords, processor.tokenizer, inputs["input_ids"])]
    else:
        stopping_criteria= None

    with torch.no_grad():
        if fea_save_path is not None:
            inputs["fea_save_path"] = fea_save_path
        output_token = model.generate(**inputs, media_type='video',
                                      do_sample=do_sample, max_new_tokens=max_new_tokens, num_beams=num_beams, min_length=min_length, 
                                      top_p=top_p, repetition_penalty=repetition_penalty, length_penalty=length_penalty, temperature=temperature, 
                                      stopping_criteria=stopping_criteria,)
        output_text = processor.batch_decode(output_token, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        if "###" in output_text:
            output_text = "###".join(output_text.split('###')[:-1])  # remove the stop sign '###'
    if print_res: # debug usage
        print('### PROMPTING LM WITH: ', prompt)
        print('### LM OUTPUT TEXT:  ', output_text)
    if conv.roles[-1] == "<|im_start|>assistant\n":
        split_tag = "<|im_start|> assistant\n"
    else:
        split_tag = conv.roles[-1]
    output_text = output_text.split(split_tag)[-1].rstrip(conv.sep if isinstance(conv.sep, str) else conv.sep[1]).strip()
    conv.messages[-1][1] = output_text
    return output_text, conv

