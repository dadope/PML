"""
Diffusion Samplers.
"""

import torch
from sampler_registry import register_sampler
from diffusion_core import DiffusionCore
from utils.helpers import extract_and_expand


@register_sampler("ddpm")
class DDPM(DiffusionCore):
    def p_sample(self, model, x, t):
        """
        Perform one reverse step of the DDPM sampler.
        """
        out = self.p_mean_variance(model, x, t)
        sample = out['mean']
        if t != 0:
            noise = torch.randn_like(x)
            sample += torch.exp(0.5 * out['log_variance']) * noise
        return {"sample": sample, "pred_xstart": out['pred_xstart']}


@register_sampler("ddim")
class DDIM(DiffusionCore):
    def p_sample(self, model, x, t, eta=0.0):
        """
        Perform one reverse step of the DDIM sampler.
        """
        out = self.p_mean_variance(model, x, t)
        eps = self.predict_eps_from_x_start(x, t, out["pred_xstart"])

        alpha_bar = extract_and_expand(self.alphas_cumprod, t, x)
        alpha_bar_prev = extract_and_expand(self.alphas_cumprod_prev, t, x)
        sigma = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
        mean_pred = out["pred_xstart"] * torch.sqrt(alpha_bar_prev) + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps

        sample = mean_pred
        if t != 0:
            noise = torch.randn_like(x)
            sample += sigma * noise

        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def predict_eps_from_x_start(self, x_t, t, pred_xstart):
        """
        Estimate epsilon from predicted x_0 and x_t.
        """
        coef1 = extract_and_expand(self.sqrt_recip_alphas_cumprod, t, x_t)
        coef2 = extract_and_expand(self.sqrt_recipm1_alphas_cumprod, t, x_t)
        return (coef1 * x_t - pred_xstart) / coef2
