"""
Sampler Registry for Diffusion Processes.
"""

import torch
import os
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from utils.img_utils import clear_color
from guided_diffusion.diffusion_core import DiffusionCore

# Registry to hold sampler classes
SAMPLER_REGISTRY = {}

def register_sampler(name: str):
	"""
	Decorator for registering a new sampler.
	"""
	def decorator(cls):
		if name in SAMPLER_REGISTRY:
			raise ValueError(f"Sampler '{name}' is already registered!")
		SAMPLER_REGISTRY[name] = cls
		return cls
	return decorator

def get_sampler(name: str):
	"""
	Retrieve a registered sampler by name.
	"""
	if name not in SAMPLER_REGISTRY:
		raise ValueError(f"Sampler '{name}' is not defined!")
	return SAMPLER_REGISTRY[name]

def create_sampler(sampler_name, diffusion_core=None, **kwargs):
	"""
	Create and initialize a sampler based on its name and parameters.

	Args:
		sampler_name (str): The name of the sampler to create.
		diffusion_core (DiffusionCore): The diffusion core to be used for the sampler.
		**kwargs: Configuration parameters for the sampler.

	Returns:
		An instance of the requested sampler.
	"""
	sampler_cls = get_sampler(sampler_name)
	return sampler_cls(diffusion_core=diffusion_core, **kwargs)

class BaseSampler:
	"""
	Base class for samplers in diffusion processes.
	"""

	def __init__(self, diffusion_core: DiffusionCore, **kwargs):
		"""
		Initialize the sampler with the diffusion core and configurations.

		Args:
			diffusion_core (DiffusionCore): Core implementation of diffusion methods.
			**kwargs: Additional configuration options.
		"""
		self.diffusion_core = diffusion_core
		self.config = kwargs

	def p_sample_loop(self, model, *args, **kwargs):
		"""
		Abstract method to define the sampling process.

		Args:
			model: The model used for sampling.
			*args, **kwargs: Arguments required for sampling.

		Returns:
			Generated samples.
		"""
		raise NotImplementedError("The 'p_sample_loop' method must be implemented in the subclass.")

@register_sampler(name="ddpm")
class DDPM(BaseSampler):
	"""
	DDPM sampler implementation.
	"""

	def __init__(self, diffusion_core: DiffusionCore, steps, **kwargs):
		super().__init__(diffusion_core, **kwargs)
		self.steps = steps
		self.num_timesteps = steps

	def p_sample_loop(self, model, x_start, measurement, measurement_cond_fn, record, save_root):
		"""
		Loop for performing sampling steps using the diffusion process.
		"""
		img = x_start
		device = x_start.device

		# Initialize progress bar
		pbar = tqdm(list(range(self.num_timesteps))[::-1], desc="Denoising Steps")
		for idx in pbar:
			time = torch.tensor([idx] * img.shape[0], device=device)

			# Generate sample using p_sample (handles adding noise, if applicable)
			out = self.diffusion_core.p_sample(model=model, x=img, t=time)

			# Extract the updated noisy image and predicted x_0
			img = out['sample']  # The noisy image after applying one reverse step
			pred_xstart = out['pred_xstart']  # The predicted denoised image

			# Apply measurement conditioning
			noisy_measurement = self.diffusion_core.q_sample(measurement, t=time)
			img, distance = measurement_cond_fn(
				x_t=img,  # Updated noisy image after sampling
				measurement=measurement,
				noisy_measurement=noisy_measurement,
				x_prev=img,
				x_0_hat=pred_xstart  # Predicted x_0
			)
			img = img.detach_()

			# Update progress bar with distance information
			pbar.set_postfix({"distance": distance.item()}, refresh=False)

			# Save intermediate images every 10 steps
			if record and idx % 10 == 0:
				save_path = os.path.join(save_root, f"step_{idx:03d}.png")  # Corrected path
				plt.imsave(save_path, clear_color(img))

		return img

