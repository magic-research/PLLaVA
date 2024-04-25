from utils.distributed import is_main_process, get_rank, get_world_size
import io
import json
import re
import numpy as np
from os.path import join
from tqdm import trange
from PIL import Image
from PIL import ImageFile
from torchvision.transforms import PILToTensor
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


def load_image_from_path(image_path, client):
    if image_path.startswith('s3') or image_path.startswith('p2'):
        value = client.Get(image_path)
        img_bytes = np.frombuffer(value, dtype=np.uint8)
        buff = io.BytesIO(img_bytes)
        image = Image.open(buff).convert('RGB')
    else:
        image = Image.open(image_path).convert('RGB')  # PIL Image
    image = PILToTensor()(image).unsqueeze(0)  # (1, C, H, W), torch.uint8
    return image

def pre_text(text, max_l=None, pre_text=True):
    if pre_text:
        text = re.sub(r"([,.'!?\"()*#:;~])", '', text.lower())
        text = text.replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

        text = re.sub(r"\s{2,}", ' ', text)
        text = text.rstrip('\n').strip(' ')

        if max_l:  # truncate
            words = text.split(' ')
            if len(words) > max_l:
                text = ' '.join(words[:max_l])
    else:
        pass
    return text

