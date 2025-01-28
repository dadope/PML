"""
Noise Registry and Implementations.
"""

from abc import ABC, abstractmethod
import torch
import numpy as np

NOISE_REGISTRY = {}

def register_noise(name: str):
    """
    Decorator for registering a noise function.
    """
    def decorator(cls):
        if name in NOISE_REGISTRY:
            raise ValueError(f"Noise '{name}' is already registered!")
        NOISE_REGISTRY[name] = cls
        return cls
    return decorator

def get_noise(name: str, **kwargs):
    """
    Retrieve a registered noise function by name.
    """
    if name not in NOISE_REGISTRY:
        raise ValueError(f"Noise '{name}' is not defined.")
    return NOISE_REGISTRY[name](**kwargs)


class Noise(ABC):
    @abstractmethod
    def forward(self, data):
        pass

    def __call__(self, data):
        return self.forward(data)


@register_noise(name="clean")
class CleanNoise(Noise):
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
        # Scale data to [0, 1] and ensure it's on the CPU for NumPy compatibility
        device = data.device
        data = (data + 1.0) / 2.0
        data = data.clamp(0, 1).cpu()

        # Apply Poisson noise using NumPy
        noisy_data = np.random.poisson(data.numpy() * 255.0 * self.rate) / (255.0 * self.rate)

        # Convert back to a PyTorch tensor and scale to [-1, 1], restoring the original device
        return torch.tensor(noisy_data, dtype=torch.float32, device=device) * 2.0 - 1.0
