"""
Core Implementation of Gaussian Diffusion.
"""

import math
import numpy as np
import torch
from utils.img_utils import clear_color
from utils.helpers import extract_and_expand, space_timesteps, get_named_beta_schedule
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import os


class DiffusionCore:
    """
    Core diffusion process with shared utilities.
    """
    def __init__(self, betas, rescale_timesteps=True):
        self.betas = np.array(betas, dtype=np.float64)
        assert self.betas.ndim == 1 and (0 < self.betas).all() and (self.betas <= 1).all()

        self.num_timesteps = len(self.betas)
        self.rescale_timesteps = rescale_timesteps

        alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(alphas)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)

    def scale_timesteps(self, t):
        """
        Rescale timesteps to a standard range if needed.
        """
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given timestep t.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        coef1 = extract_and_expand(self.sqrt_alphas_cumprod, t, x_start)
        coef2 = extract_and_expand(self.sqrt_one_minus_alphas_cumprod, t, x_start)
        return coef1 * x_start + coef2 * noise
