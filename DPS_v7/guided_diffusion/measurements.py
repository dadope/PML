"""
This module handles task-dependent operations (A) and noises (n) to simulate a measurement y=Ax+n.
"""

from abc import ABC, abstractmethod
from functools import partial
import yaml
from torch.nn import functional as F
import torch
from motionblur.motionblur import Kernel
from utils.resizer import Resizer
from utils.img_utils import Blurkernel, fft2_m
import numpy as np
from abc import ABC, abstractmethod


# =================
# Operation classes
# =================

__OPERATOR__ = {}

def register_operator(name: str):
    def wrapper(cls):
        if __OPERATOR__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __OPERATOR__[name] = cls
        return cls
    return wrapper


def get_operator(name: str, **kwargs):
    if __OPERATOR__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __OPERATOR__[name](**kwargs)

class Noise(ABC):
    """
    Abstract base class for noise implementations.
    """
    @abstractmethod
    def forward(self, data):
        pass

    def __call__(self, data):
        return self.forward(data)

class LinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        pass

    @abstractmethod
    def transpose(self, data, **kwargs):
        pass

    def ortho_project(self, data, **kwargs):
        return data - self.transpose(self.forward(data, **kwargs), **kwargs)

    def project(self, data, measurement, **kwargs):
        return self.ortho_project(measurement, **kwargs) - self.forward(data, **kwargs)


@register_operator(name="super_resolution")
class SuperResolutionOperator(LinearOperator):
    def __init__(self, in_shape, scale_factor, device):
        self.device = device
        self.up_sample = partial(F.interpolate, scale_factor=scale_factor)
        self.down_sample = Resizer(in_shape, 1 / scale_factor).to(device)

    def forward(self, data, **kwargs):
        return self.down_sample(data.to(self.device))

    def transpose(self, data, **kwargs):
        return self.up_sample(data.to(self.device))


@register_operator(name="motion_blur")
class MotionBlurOperator(LinearOperator):
    def __init__(self, kernel_size, intensity, device):
        self.device = device
        self.kernel_size = kernel_size
        self.conv = Blurkernel(
            blur_type="motion", kernel_size=kernel_size, std=intensity, device=device
        ).to(device)

        self.kernel = Kernel(size=(kernel_size, kernel_size), intensity=intensity)
        kernel_tensor = torch.tensor(self.kernel.kernelMatrix, dtype=torch.float32, device=device)
        self.conv.update_weights(kernel_tensor)

    def forward(self, data, **kwargs):
        return self.conv(data.to(self.device))

    def transpose(self, data, **kwargs):
        return data.to(self.device)


@register_operator(name="gaussian_blur")
class GaussianBlurOperator(LinearOperator):
    def __init__(self, kernel_size, intensity, device):
        self.device = device
        self.kernel_size = kernel_size
        self.conv = Blurkernel(
            blur_type="gaussian", kernel_size=kernel_size, std=intensity, device=device
        ).to(device)

    def forward(self, data, **kwargs):
        return self.conv(data.to(self.device))

    def transpose(self, data, **kwargs):
        return data.to(self.device)


@register_operator(name="inpainting")
class InpaintingOperator(LinearOperator):
    def __init__(self, device):
        self.device = device

    def forward(self, data, **kwargs):
        mask = kwargs.get("mask", None)
        if mask is None:
            raise ValueError("Mask is required for inpainting.")
        return data.to(self.device) * mask.to(self.device)

    def transpose(self, data, **kwargs):
        return data.to(self.device)


@register_operator(name="phase_retrieval")
class PhaseRetrievalOperator(LinearOperator):
    def __init__(self, oversample, device):
        self.pad = int((oversample / 8.0) * 256)
        self.device = device

    def forward(self, data, **kwargs):
        padded = F.pad(data.to(self.device), (self.pad, self.pad, self.pad, self.pad))
        amplitude = fft2_m(padded).abs()
        return amplitude


@register_operator(name="nonlinear_blur")
class NonlinearBlurOperator(LinearOperator):
    def __init__(self, opt_yml_path, device):
        self.device = device
        self.blur_model = self.prepare_nonlinear_blur_model(opt_yml_path)

    def prepare_nonlinear_blur_model(self, opt_yml_path):
        from bkse.models.kernel_encoding.kernel_wizard import KernelWizard

        with open(opt_yml_path, "r") as f:
            opt = yaml.safe_load(f)["KernelWizard"]
            model_path = opt["pretrained"]
        blur_model = KernelWizard(opt)
        blur_model.eval()
        blur_model.load_state_dict(torch.load(model_path, map_location=self.device))
        return blur_model.to(self.device)

    def forward(self, data, **kwargs):
        random_kernel = torch.randn(1, 512, 2, 2, device=self.device) * 1.2
        data = (data + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        blurred = self.blur_model.adaptKernel(data.to(self.device), kernel=random_kernel)
        blurred = (blurred * 2.0 - 1.0).clamp(-1, 1)  # [0, 1] -> [-1, 1]
        return blurred


# =============
# Noise classes
# =============

__NOISE__ = {}

def register_noise(name: str):
    def wrapper(cls):
        if __NOISE__.get(name, None):
            raise NameError(f"Name {name} is already defined!")
        __NOISE__[name] = cls
        return cls
    return wrapper


def get_noise(name: str, **kwargs):
    if __NOISE__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    noiser = __NOISE__[name](**kwargs)
    noiser.__name__ = name
    return noiser


@register_noise(name="clean")
class Clean(Noise):
    def forward(self, data):
        return data


@register_noise(name="gaussian")
class GaussianNoise(Noise):
    def __init__(self, sigma):
        self.sigma = sigma

    def forward(self, data):
        return data + torch.randn_like(data, device=data.device) * self.sigma


@register_noise(name="poisson")
class PoissonNoise(Noise):
    def __init__(self, rate):
        self.rate = rate

    def forward(self, data):
        device = data.device
        data = (data + 1.0) / 2.0
        data = data.clamp(0, 1).cpu()
        noisy_data = np.random.poisson(data.numpy() * 255.0 * self.rate) / (255.0 * self.rate)
        noisy_data = torch.tensor(noisy_data, dtype=torch.float32).to(device)
        return noisy_data * 2.0 - 1.0
