import torch
from abc import ABC, abstractmethod
from utils.img_utils import fft2, ifft2
from guided_diffusion.noise_registry import GaussianNoise, PoissonNoise
from torch import linalg as LA
# Registry for conditioning methods
CONDITIONING_REGISTRY = {}

def register_conditioning(name: str):
    """
    Register a conditioning class by name.
    """
    def decorator(cls):
        if name in CONDITIONING_REGISTRY:
            raise ValueError(f"Conditioning method '{name}' is already registered.")
        CONDITIONING_REGISTRY[name] = cls
        return cls
    return decorator

def get_conditioning(name: str, operator, noiser, **kwargs):
    """
    Retrieve and initialize a conditioning method by its name.
    """
    if name not in CONDITIONING_REGISTRY:
        raise ValueError(f"Conditioning method '{name}' is not registered.")
    return CONDITIONING_REGISTRY[name](operator=operator, noiser=noiser, **kwargs)

class BaseConditioning(ABC):
    """
    Base class for all conditioning methods.
    """
    def __init__(self, operator, noiser, **kwargs):
        self.operator = operator
        self.noiser = noiser

    def apply_projection(self, data, noisy_measurement, **kwargs):
        """
        Project data using the operator.
        """
        data = data.to(noisy_measurement.device)
        return self.operator.project(data, noisy_measurement, **kwargs)

    def compute_gradient_and_norm(self, x_prev, x_0_hat, measurement, **kwargs):
        """
        Compute the gradient and norm based on the noise type.
        """
        x_prev = x_prev.to(measurement.device).requires_grad_(True)

        if isinstance(self.noiser, GaussianNoise):
            diff = measurement - self.operator.forward(x_prev, **kwargs)
        elif isinstance(self.noiser, PoissonNoise):
            Ax = self.operator.forward(x_prev, **kwargs)
            diff = (measurement - Ax) / (measurement.abs().mean() + 1e-8)
        else:
            raise NotImplementedError(f"Unsupported noise type: {type(self.noiser).__name__}")

        norm = LA.vector_norm(diff)
        gradient = torch.autograd.grad(
            outputs=norm, inputs=x_prev, retain_graph=False, allow_unused=True
        )[0]

        if gradient is None:
            raise RuntimeError("Gradient is None. Ensure computation is connected to the graph.")

        return gradient.detach(), norm.detach()

    @abstractmethod
    def condition(self, x_prev, x_t, measurement, **kwargs):
        """
        Abstract method for conditioning logic.
        """
        pass

@register_conditioning(name="ps")
class PhaseRetrievalConditioning(BaseConditioning):
    """
    Phase retrieval (ps) conditioning method.
    """
    def __init__(self, operator, noiser, scale=1, **kwargs):
        super().__init__(operator, noiser, **kwargs)
        self.scale = scale

    def condition(self, x_prev, x_t, measurement, **kwargs):
        """
        Apply phase retrieval conditioning.
        """
        x_prev = x_prev.to(measurement.device).requires_grad_(True)
        x_t = x_t.to(measurement.device)
        x_0_hat = kwargs.pop('x_0_hat', None)
        if x_0_hat is not None:
            x_0_hat = x_0_hat.to(measurement.device)

        gradient, norm = self.compute_gradient_and_norm(x_prev, x_0_hat, measurement, **kwargs)
        x_t = x_t - self.scale*norm * gradient
        return x_t.detach(), norm.detach()