"""
Linear Operators for Measurements.
"""

from abc import ABC, abstractmethod
from functools import partial
from utils.resizer import Resizer
from utils.img_utils import Blurkernel
import torch
from motionblur.motionblur import Kernel

REGISTERED_OPERATORS = {}

def register_operator(name: str):
    """
    Decorator to register a LinearOperator class with a given name.
    """
    def wrapper(cls):
        if name in REGISTERED_OPERATORS:
            raise ValueError(f"Operator {name} is already registered.")
        REGISTERED_OPERATORS[name] = cls
        return cls
    return wrapper

def get_operator(name: str, **kwargs):
    """
    Retrieve a registered operator by name.
    """
    if name not in REGISTERED_OPERATORS:
        raise ValueError(f"Operator {name} is not registered.")
    return REGISTERED_OPERATORS[name](**kwargs)


class LinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        pass

    @abstractmethod
    def transpose(self, data, **kwargs):
        pass

    def project(self, data, measurement, **kwargs):
        """
        Orthogonal projection of data based on measurements.
        """
        data = data.to(measurement.device)
        return self.ortho_project(measurement, **kwargs) - self.forward(data, **kwargs)

    def ortho_project(self, data, **kwargs):
        """
        Orthogonal projection operation.
        """
        data = data.to(self.device)
        return data - self.transpose(self.forward(data, **kwargs), **kwargs)


@register_operator(name="super_resolution")
class SuperResolutionOperator(LinearOperator):
    def __init__(self, in_shape, scale_factor, device):
        self.device = device
        self.up_sample = partial(torch.nn.functional.interpolate, scale_factor=scale_factor)
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
        self.conv = Blurkernel(blur_type="motion", kernel_size=kernel_size, std=intensity, device=device).to(device)
        self.kernel = Kernel(size=(kernel_size, kernel_size), intensity=intensity).kernelMatrix
        self.conv.update_weights(torch.tensor(self.kernel, dtype=torch.float32).to(device))

    def forward(self, data, **kwargs):
        return self.conv(data.to(self.device))

    def transpose(self, data, **kwargs):
        return data.to(self.device)


@register_operator(name="inpainting")
class InpaintingOperator(LinearOperator):
    def __init__(self, image_size, mask_generator, device):
        """
        Initialize the InpaintingOperator.

        Args:
            image_size (int): The size of the input image (e.g., 256 for 256x256 images).
            mask_generator (callable): A function to generate masks.
            device (torch.device): The device to run the operator on.
        """
        self.image_size = image_size
        self.mask_generator = mask_generator
        self.device = device

    def forward(self, data, mask=None, **kwargs):
        """
        Apply the inpainting operator (masking).

        Args:
            data (torch.Tensor): The input image tensor.
            mask (torch.Tensor, optional): The mask to apply. If None, a mask will be generated.

        Returns:
            torch.Tensor: The masked image.
        """
        data = data.to(self.device)
        if mask is None:
            mask = self.mask_generator(data).to(self.device)
        return data * mask

    def transpose(self, data, mask=None, **kwargs):
        """
        Apply the transpose of the inpainting operator.

        Args:
            data (torch.Tensor): The input image tensor.
            mask (torch.Tensor, optional): The mask to apply. If None, a mask will be generated.

        Returns:
            torch.Tensor: The masked image (same as forward for inpainting).
        """
        data = data.to(self.device)
        if mask is None:
            mask = self.mask_generator(data).to(self.device)
        return data * mask


@register_operator(name="gaussian_blur")
class GaussianBlurOperator(LinearOperator):
    def __init__(self, kernel_size, sigma, device):
        """
        Gaussian Blur Operator.

        :param kernel_size: Size of the Gaussian kernel.
        :param sigma: Standard deviation of the Gaussian blur.
        :param device: Device to run the operation on (CPU/GPU).
        """
        self.device = device
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.conv = Blurkernel(
            blur_type="gaussian", 
            kernel_size=kernel_size, 
            std=sigma, 
            device=device
        ).to(device)

    def forward(self, data, **kwargs):
        """
        Apply Gaussian blur.
        """
        return self.conv(data.to(self.device))

    def transpose(self, data, **kwargs):
        """
        Transpose operation for Gaussian blur (equivalent to forward).
        """
        return self.forward(data)

