"""
Core Implementation of Gaussian Diffusion.
"""

import torch
from utils.helpers import extract_and_expand
import numpy as np


class DiffusionCore:
	"""
	Core diffusion process with shared utilities.
	"""

	def __init__(self, betas, rescale_timesteps=True, device=None):
		self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.betas = torch.tensor(betas, dtype=torch.float64, device=self.device)
		assert self.betas.ndim == 1 and (0 < self.betas).all() and (self.betas <= 1).all()

		self.num_timesteps = len(self.betas)
		self.rescale_timesteps = rescale_timesteps

		alphas = 1.0 - self.betas
		self.alphas_cumprod = torch.cumprod(alphas, dim=0).to(self.device)
		self.alphas_cumprod_prev = torch.cat(
			(torch.tensor([1.0], device=self.device), self.alphas_cumprod[:-1])
		)
		self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(self.device)
		self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod).to(self.device)
		self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod).to(self.device)
		self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1).to(self.device)
		self.posterior_variance = (
			self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
		)
		self.posterior_log_variance = torch.log(self.posterior_variance).to(self.device)
		self.posterior_mean_coef1 = (
			self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
		)
		self.posterior_mean_coef2 = (
			(1.0 - self.alphas_cumprod_prev)
			* torch.sqrt(alphas)
			/ (1.0 - self.alphas_cumprod)
		)

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
		x_start = x_start.to(self.device)
		t = t.to(self.device)
		if noise is None:
			noise = torch.randn_like(x_start, device=self.device)
		else:
			noise = noise.to(self.device)

		coef1 = extract_and_expand(self.sqrt_alphas_cumprod, t, x_start)
		coef2 = extract_and_expand(self.sqrt_one_minus_alphas_cumprod, t, x_start)
		return coef1 * x_start + coef2 * noise

	def q_posterior_mean_variance(self, x_start, x_t, t):
		"""
		Compute the posterior mean and variance q(x_{t-1} | x_t, x_0).
		"""
		coef1 = extract_and_expand(self.posterior_mean_coef1, t, x_start)
		coef2 = extract_and_expand(self.posterior_mean_coef2, t, x_t)
		posterior_mean = coef1 * x_start + coef2 * x_t
		posterior_variance = extract_and_expand(self.posterior_variance, t, x_t)
		posterior_log_variance = extract_and_expand(self.posterior_log_variance, t, x_t)
		return posterior_mean, posterior_variance, posterior_log_variance

	def predict_xstart_from_eps(self, x_t, t, eps):
		"""
		Predict the denoised image (x_0) from the noisy image (x_t) and the predicted noise (eps).
		"""
		coef1 = extract_and_expand(self.sqrt_recip_alphas_cumprod, t, x_t)
		coef2 = extract_and_expand(self.sqrt_recipm1_alphas_cumprod, t, x_t)
		return coef1 * x_t - coef2 * eps

	def p_mean_variance(self, model, x, t):
		"""
		Compute the predicted mean and variance for the current timestep.
		"""
		model_output = model(x, self.scale_timesteps(t))

		# Handle learned variance if the model predicts it
		if model_output.shape[1] == 2 * x.shape[1]:
			model_output, model_var_values = torch.split(model_output, x.shape[1], dim=1)
		else:
			model_var_values = None

		pred_xstart = self.predict_xstart_from_eps(x, t, model_output)
		mean, variance, log_variance = self.q_posterior_mean_variance(pred_xstart, x, t)

		return {
			"mean": mean,
			"variance": variance,
			"log_variance": log_variance,
			"pred_xstart": pred_xstart,
		}

	def p_sample(self, model, x, t):
		"""
		Sample one step from the reverse diffusion process.
		"""
		out = self.p_mean_variance(model, x, t)

		# Add noise unless at the last step
		noise = torch.randn_like(x) if t[0] > 0 else torch.zeros_like(x)
		sample = out["mean"] + torch.exp(0.5 * out["log_variance"]) * noise

		return {
			"sample": sample,  # The current noisy image
			"pred_xstart": out["pred_xstart"],  # The predicted denoised image
		}

