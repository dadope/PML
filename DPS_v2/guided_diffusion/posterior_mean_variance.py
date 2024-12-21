from abc import ABC, abstractmethod
import numpy as np
import torch
from utils.img_utils import dynamic_thresholding

# Registry utilities for mean processors
__MODEL_MEAN_PROCESSOR__ = {}

def register_mean_processor(name: str):
    def wrapper(cls):
        if name in __MODEL_MEAN_PROCESSOR__:
            raise NameError(f"Mean Processor {name} is already registered.")
        __MODEL_MEAN_PROCESSOR__[name] = cls
        return cls
    return wrapper

def get_mean_processor(name: str, **kwargs):
    if name not in __MODEL_MEAN_PROCESSOR__:
        raise NameError(f"Mean Processor {name} is not defined.")
    return __MODEL_MEAN_PROCESSOR__[name](**kwargs)

# Registry utilities for variance processors
__MODEL_VAR_PROCESSOR__ = {}

def register_var_processor(name: str):
    def wrapper(cls):
        if name in __MODEL_VAR_PROCESSOR__:
            raise NameError(f"Variance Processor {name} is already registered.")
        __MODEL_VAR_PROCESSOR__[name] = cls
        return cls
    return wrapper

def get_var_processor(name: str, **kwargs):
    if name not in __MODEL_VAR_PROCESSOR__:
        raise NameError(f"Variance Processor {name} is not defined.")
    return __MODEL_VAR_PROCESSOR__[name](**kwargs)

class MeanProcessor(ABC):
    """
    Base class for predicting x_start and calculating mean values.
    """
    def __init__(self, betas, dynamic_threshold, clip_denoised):
        self.dynamic_threshold = dynamic_threshold
        self.clip_denoised = clip_denoised

    @abstractmethod
    def get_mean_and_xstart(self, x, t, model_output):
        pass

    def process_xstart(self, x):
        if self.dynamic_threshold:
            x = dynamic_thresholding(x, s=0.95)
        if self.clip_denoised:
            x = x.clamp(-1, 1)
        return x


class VarianceProcessor(ABC):
    """
    Base class for variance calculation.
    """
    def __init__(self, betas):
        pass

    @abstractmethod
    def get_variance(self, x, t):
        pass


@register_mean_processor(name='previous_x')
class PreviousXMeanProcessor(MeanProcessor):
    def __init__(self, betas, dynamic_threshold, clip_denoised):
        super().__init__(betas, dynamic_threshold, clip_denoised)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        self.posterior_mean_coef1 = betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)

    def predict_xstart(self, x_t, t, x_prev):
        coef1 = extract_and_expand(1.0 / self.posterior_mean_coef1, t, x_t)
        coef2 = extract_and_expand(self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t)
        return coef1 * x_prev - coef2 * x_t

    def get_mean_and_xstart(self, x, t, model_output):
        mean = model_output
        pred_xstart = self.process_xstart(self.predict_xstart(x, t, model_output))
        return mean, pred_xstart


@register_mean_processor(name='start_x')
class StartXMeanProcessor(MeanProcessor):
    def __init__(self, betas, dynamic_threshold, clip_denoised):
        super().__init__(betas, dynamic_threshold, clip_denoised)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        self.posterior_mean_coef1 = betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)

    def q_posterior_mean(self, x_start, x_t, t):
        coef1 = extract_and_expand(self.posterior_mean_coef1, t, x_start)
        coef2 = extract_and_expand(self.posterior_mean_coef2, t, x_t)
        return coef1 * x_start + coef2 * x_t

    def get_mean_and_xstart(self, x, t, model_output):
        pred_xstart = self.process_xstart(model_output)
        mean = self.q_posterior_mean(pred_xstart, x, t)
        return mean, pred_xstart

@register_var_processor(name='fixed_small')
class FixedSmallVarianceProcessor(VarianceProcessor):
    def __init__(self, betas):
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        self.posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

    def get_variance(self, x, t):
        model_variance = extract_and_expand(self.posterior_variance, t, x)
        model_log_variance = extract_and_expand(np.log(self.posterior_variance), t, x)
        return model_variance, model_log_variance


@register_var_processor(name='learned')
class LearnedVarianceProcessor(VarianceProcessor):
    def get_variance(self, x, t):
        model_log_variance = x
        model_variance = torch.exp(model_log_variance)
        return model_variance, model_log_variance

def extract_and_expand(array, time, target):
    array = torch.from_numpy(array).to(target.device)[time].float()
    while array.ndim < target.ndim:
        array = array.unsqueeze(-1)
    return array.expand_as(target)
