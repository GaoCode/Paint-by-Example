import os
import torch
import numpy as np
from PIL import Image

from imwatermark import WatermarkEncoder

from einops import rearrange

import torchvision


from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from transformers import AutoFeatureExtractor

wm = "Paint-by-Example"
wm_encoder = WatermarkEncoder()
wm_encoder.set_watermark("bytes", wm.encode("utf-8"))
safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)


def get_tensor(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    return torchvision.transforms.Compose(transform_list)


def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [
            torchvision.transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            )
        ]
    return torchvision.transforms.Compose(transform_list)


def un_norm(x):
    return (x + 1.0) / 2.0


def un_norm_clip(x):
    x[0, :, :] = x[0, :, :] * 0.26862954 + 0.48145466
    x[1, :, :] = x[1, :, :] * 0.26130258 + 0.4578275
    x[2, :, :] = x[2, :, :] * 0.27577711 + 0.40821073
    return x


image_path = "examples/noosa/1559167625_+00240.jpg"
filename = os.path.basename(image_path)
img_p = Image.open(image_path).convert("RGB")
image_tensor = get_tensor()(img_p)
image_tensor = image_tensor.unsqueeze(0)

mask_path = "examples/noosa/1559167625_+00240.png"
mask = Image.open(mask_path).convert("L")
mask = np.array(mask)[None, None]
mask = 1 - mask.astype(np.float32) / 255.0
mask[mask < 0.5] = 0
mask[mask >= 0.5] = 1
mask_tensor = torch.from_numpy(mask)

inpaint_image = image_tensor * mask_tensor
print(image_tensor.shape, mask_tensor.shape, inpaint_image.shape)


inpaint_image = torch.squeeze(inpaint_image)

sample_path = "examples/noosa/reference.jpg"
inpaint_img = 255.0 * rearrange(inpaint_image, "c h w -> h w c").numpy()
inpaint_img = Image.fromarray(inpaint_img.astype(np.uint8))
inpaint_img.save(sample_path)
