import os
import argparse
import yaml
from functools import partial

import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from utils.logger import get_logger
from utils.img_utils import clear_color, mask_generator
from data.dataloader import get_dataset, get_dataloader
from guided_diffusion.linear_operators import get_operator
from guided_diffusion.noise_registry import get_noise
from guided_diffusion.conditioning_registry import get_conditioning
from guided_diffusion.unet import create_model
from guided_diffusion.sampler_registry import create_sampler
from guided_diffusion.nonlinear_operators import *



def load_yaml(file_path: str) -> dict:
	with open(file_path, 'r') as f:
		return yaml.safe_load(f)


def prepare_output_dirs(base_dir, sub_dirs):
	"""Prepare output directories for saving results."""
	os.makedirs(base_dir, exist_ok=True)
	for sub_dir in sub_dirs:
		os.makedirs(os.path.join(base_dir, sub_dir), exist_ok=True)


def main():
	parser = argparse.ArgumentParser(description="Run diffusion-based image restoration with conditioning.")
	parser.add_argument('--model_config', type=str, required=True, help="Path to the model configuration YAML.")
	parser.add_argument('--diffusion_config', type=str, required=True, help="Path to the diffusion configuration YAML.")
	parser.add_argument('--task_config', type=str, required=True, help="Path to the task configuration YAML.")
	parser.add_argument('--gpu', type=int, default=0, help="GPU index for computation.")
	parser.add_argument('--save_dir', type=str, default='./results', help="Directory to save outputs.")
	args = parser.parse_args()

	# Logger setup
	logger = get_logger()

	# Device setup
	device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
	logger.info(f"Using device: {device}")

	# Load configurations
	model_config = load_yaml(args.model_config)
	diffusion_config = load_yaml(args.diffusion_config)
	task_config = load_yaml(args.task_config)

	# Initialize model
	model = create_model(**model_config).to(device)
	model.eval()
	logger.info("Model loaded and set to evaluation mode.")

	# Prepare operator and noise
	measure_config = task_config['measurement']
	operator_name = measure_config['operator']['name']
	mask_gen = None

	# Initialize operator based on the task
	operator_args = measure_config['operator']
	if 'mask_opt' in measure_config:  # For tasks like inpainting
		mask_gen = mask_generator(**measure_config['mask_opt'])
		operator_args['mask_generator'] = mask_gen

	# Get the operator
	operator = get_operator(device=device, **operator_args)

	noiser = get_noise(**measure_config['noise'])
	logger.info(f"Operator: {operator_name}, Noise: {measure_config['noise']['name']}")

	# Prepare conditioning method
	cond_config = task_config['conditioning']
	cond_method = get_conditioning(cond_config['method'], operator, noiser, **cond_config['params'])
	measurement_cond_fn = cond_method.condition
	logger.info(f"Conditioning method: {cond_config['method']}")

	# Load sampler
	sampler = create_sampler(**diffusion_config)
	sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=measurement_cond_fn)

	# Prepare output directories
	output_dirs = ["input", "label", "recon", "progress"]
	out_path = os.path.join(args.save_dir, operator_name)
	prepare_output_dirs(out_path, output_dirs)

	# Prepare data loader
	data_config = task_config['data']
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])
	dataset = get_dataset(**data_config, transforms=transform)
	loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)

	# Perform inference
	for idx, ref_img in enumerate(loader):
		logger.info(f"Processing image {idx}...")
		ref_img = ref_img.to(device)
		fname = f"{str(idx).zfill(5)}.png"

		# Handle operator-specific requirements
		if operator_name == 'inpainting':
			mask = mask_gen(ref_img).to(device)
			mask = mask[:, 0, :, :].unsqueeze(1)
			measurement_cond_fn = partial(cond_method.condition, mask=mask)
			sample_fn = partial(sample_fn, measurement_cond_fn=measurement_cond_fn)
			y = operator.forward(ref_img, mask=mask)
		else:
			y = operator.forward(ref_img)
		
		y_n = noiser(y)
		x_start = torch.randn(ref_img.shape, device=device, requires_grad=True)
		sample = sample_fn(x_start=x_start, measurement=y_n, record=True, save_root=out_path)

		# Save results
		plt.imsave(os.path.join(out_path, "input", fname), clear_color(y_n))
		plt.imsave(os.path.join(out_path, "label", fname), clear_color(ref_img))
		plt.imsave(os.path.join(out_path, "recon", fname), clear_color(sample))

if __name__ == "__main__":
	main()
