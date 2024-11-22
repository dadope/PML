from enum import Enum

import torch
import numpy as np
from PIL import Image
from torchvision.transforms import transforms, v2, InterpolationMode

from bin.motion_blur import MotionBlur


# taken from: https://github.com/DPS2022/diffusion-posterior-sampling/blob/main/util/tools.py
def normalize_np(img):
    img -= np.min(img)
    img = np.divide(img, np.max(img))
    return Image.fromarray(img.astype('uint8'), 'RGB')

def center_crop_necessary(img):
    width, height = img.size
    if width > 256 and height > 256:
        img = transforms.CenterCrop((256, 256))(img)

    return img


DEFAULT_AUGMENTATIONS = [
    #transforms.Resize((256, 256)),
    #transforms.RandomHorizontalFlip(p=0.5),
    #transforms.RandomVerticalFlip(p=0.5),
    center_crop_necessary,
    #transforms.CenterCrop((256, 256)),
    v2.GaussianBlur(kernel_size=(61, 61), sigma=0.005),
]


class RandomInpaint(v2.Transform):
    def __init__(self, p=0.9):
        super().__init__()
        self.p = p

    def forward(self, img):
        mask = torch.rand(1, img.size(1), img.size(2)) > self.p
        return img * mask


tensor_to_pil = transforms.ToPILImage()
pil_to_tensor = transforms.PILToTensor()

pil_transform = lambda transform: transforms.Compose([*DEFAULT_AUGMENTATIONS, transform, pil_to_tensor])
tensor_transform = lambda transform: transforms.Compose([*DEFAULT_AUGMENTATIONS, pil_to_tensor, transform])

_non_uniform_transform = lambda image: torch.log(image + 0.00001)
_fourier_magnitute = lambda image: torch.abs(torch.fft.fftn(image))


class ImageTransforms(Enum):
    block_inpaint = tensor_transform(transforms.RandomErasing(p=1.0))
    downscale_4x = pil_transform(v2.Resize(size=64, interpolation=InterpolationMode.BICUBIC))
    downscale_16x = pil_transform(v2.Resize(size=16, interpolation=InterpolationMode.BICUBIC))
    gaussian_blur = pil_transform(v2.GaussianBlur(kernel_size=(61, 61), sigma=3.0))
    motion_blur = pil_transform(MotionBlur())
    random_inpaint = tensor_transform(RandomInpaint())
    fourier_phase_transform = tensor_transform(_fourier_magnitute)
    non_uniform = tensor_transform(_non_uniform_transform)

    @classmethod
    def get_transforms(cls):
        return {transform_name: transform.value for transform_name, transform in cls.__members__.items()}
