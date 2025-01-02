"""
Nonlinear Operators for Measurements.
"""

from abc import ABC, abstractmethod
from utils.img_utils import fft2_m
from torch.nn import functional as F
import torch
import yaml
from guided_diffusion.linear_operators import register_operator, REGISTERED_OPERATORS

class NonLinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        pass

    def project(self, data, measurement, **kwargs):
        """
        Project data using measurement for nonlinear operations.
        """
        return data + measurement - self.forward(data)


@register_operator(name="phase_retrieval")
class PhaseRetrievalOperator(NonLinearOperator):
    def __init__(self, oversample, device):
        self.pad = int((oversample / 8.0) * 256)
        self.device = device

    def forward(self, data, **kwargs):
        # Perform forward operation: FFT and compute amplitude
        padded = F.pad(data, (self.pad, self.pad, self.pad, self.pad))
        amplitude = fft2_m(padded).abs()
        return amplitude

    def transpose(self, measurement, **kwargs):
        # Transpose operation for phase retrieval (placeholder)
        # It typically involves an inverse operation but can return measurement directly for conditioning
        return measurement


@register_operator(name="nonlinear_blur")
class NonlinearBlurOperator(NonLinearOperator):
    def __init__(self, opt_yml_path, device):
        self.device = device
        self.blur_model = self.prepare_nonlinear_blur_model(opt_yml_path)

    def prepare_nonlinear_blur_model(self, opt_yml_path):
        """
        Load and prepare a nonlinear blur model.
        """
        from bkse.models.kernel_encoding.kernel_wizard import KernelWizard

        with open(opt_yml_path, "r") as f:
            opt = yaml.safe_load(f)["KernelWizard"]
            model_path = opt["pretrained"]
        model = KernelWizard(opt)
        model.load_state_dict(torch.load(model_path))
        model.eval().to(self.device)
        return model

    def forward(self, data, **kwargs):
        random_kernel = torch.randn(1, 512, 2, 2).to(self.device) * 1.2
        data = (data + 1.0) / 2.0
        blurred = self.blur_model.adaptKernel(data, kernel=random_kernel)
        return (blurred * 2.0 - 1.0).clamp(-1, 1)
