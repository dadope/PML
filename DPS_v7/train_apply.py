import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import yaml
from functools import partial
import time
import numpy as np


from guided_diffusion.unet import create_model
from guided_diffusion.diffusion_core import DiffusionCore
from guided_diffusion.sampler_registry import create_sampler
from guided_diffusion.linear_operators import get_operator
from guided_diffusion.noise_registry import get_noise
from guided_diffusion.conditioning_registry import get_conditioning
from guided_diffusion.nonlinear_operators import *
from data.dataloader import get_dataset, get_dataloader
from utils.logger import get_logger
from utils.img_utils import clear_color, mask_generator
from torch.utils.data import Dataset

# Automatically select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class CustomImageDataset(Dataset):
	def __init__(self, root_dir, transform=None):
		self.root_dir = root_dir
		self.file_names = [f for f in os.listdir(root_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
		self.transform = transform

	def __len__(self):
		return len(self.file_names)

	def __getitem__(self, idx):
		img_path = os.path.join(self.root_dir, self.file_names[idx])
		image = Image.open(img_path).convert("RGB")
		if self.transform:
			image = self.transform(image)
		return image


def load_yaml(file_path: str) -> dict:
	with open(file_path, 'r') as f:
		return yaml.safe_load(f)


def prepare_output_dirs(base_dir, sub_dirs):
	os.makedirs(base_dir, exist_ok=True)
	for sub_dir in sub_dirs:
		os.makedirs(os.path.join(base_dir, sub_dir), exist_ok=True)
  
def train_model(model, diffusion, train_loader, val_loader, epochs, optimizer, scheduler, criterion, save_dir):
	logger = get_logger()
	logger.info("Starting training...")
	
	# Track loss history
	loss_history = []
	val_loss_history = []

	# Create directory for saving checkpoints if not already present
	os.makedirs(save_dir, exist_ok=True)

	# Loop over epochs
	for epoch in range(epochs):
		start_time = time.time()
		model.train()
		epoch_loss = 0

		# Training phase
		for inputs in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"):
			inputs = inputs.to(device).requires_grad_(True)  # Ensure inputs require gradients
			optimizer.zero_grad()  # Reset gradients

			t = torch.randint(0, diffusion.num_timesteps, (inputs.size(0),), device=device)  # Random timesteps
			noise = torch.randn_like(inputs, device=device)  # Gaussian noise
			noisy_inputs = diffusion.q_sample(inputs, t, noise=noise)

			# Forward pass
			outputs = model(noisy_inputs, t)  # Model outputs
			loss = criterion(outputs, noise)  # Calculate loss

			# Backward pass and optimizer step
			loss.backward()  # Compute gradients
			optimizer.step()  # Update model parameters
			epoch_loss += loss.item()  # Accumulate loss

		scheduler.step()  # Adjust learning rate

		# Calculate average epoch loss
		avg_epoch_loss = epoch_loss / len(train_loader)
		loss_history.append(avg_epoch_loss)

		logger.info(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_epoch_loss:.4f}")

		# Validation phase (if validation loader is provided)
		if val_loader:
			model.eval()
			val_loss = 0
			with torch.no_grad():  # Disable gradient computation
				for inputs in val_loader:
					inputs = inputs.to(device)

					t = torch.randint(0, diffusion.num_timesteps, (inputs.size(0),), device=device)
					noise = torch.randn_like(inputs, device=device)
					noisy_inputs = diffusion.q_sample(inputs, t, noise=noise)

					outputs = model(noisy_inputs, t)
					val_loss += criterion(outputs, noise).item()

			avg_val_loss = val_loss / len(val_loader)
			val_loss_history.append(avg_val_loss)
			logger.info(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {avg_val_loss:.4f}")

		# Define the models directory
		models_dir = "./models"
		os.makedirs(models_dir, exist_ok=True)  # Ensure the models directory exists

		# Save checkpoint in the models directory
		torch.save(model.state_dict(), f"{models_dir}/model_epoch_{epoch + 1}.pt")
		logger.info(f"Saved model checkpoint at epoch {epoch + 1} in {models_dir}")


		# Log epoch time
		epoch_time = time.time() - start_time
		logger.info(f"Epoch {epoch + 1} completed in {epoch_time:.2f} seconds.")

	# Plot and save training and validation loss curves
	plt.figure()
	plt.plot(loss_history, label="Training Loss")
	if val_loader:
		plt.plot(val_loss_history, label="Validation Loss")
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.legend()
	plt.title("Training and Validation Loss")
	plt.savefig(f"{save_dir}/loss_plot.png")
	plt.close()
	logger.info(f"Training completed. Loss plots saved to {save_dir}/loss_plot.png")
def perform_inference(configs, task_config, model_checkpoint, save_dir):
	logger = get_logger()
	logger.info("Starting inference...")

	# Load configurations
	model_config = load_yaml(configs['model_config'])
	diffusion_config = load_yaml(configs['diffusion_config'])
	task_config = load_yaml(task_config)

	# Initialize model
	model = create_model(**model_config).to(device)
	checkpoint = torch.load(model_checkpoint, map_location=device)  # Load checkpoint
	model.load_state_dict(checkpoint, strict=False)  # Load weights
	model.eval()

	# Initialize diffusion process
	valid_diffusion_args = {"betas", "rescale_timesteps"}
	diffusion_args = {k: v for k, v in diffusion_config.items() if k in valid_diffusion_args}

	# Provide a default beta schedule if not specified
	if "betas" not in diffusion_args or diffusion_args["betas"] == "...":
		diffusion_args["betas"] = np.linspace(1e-4, 0.02, 1000)  # Default linear beta schedule

	diffusion = DiffusionCore(**diffusion_args)

	# Prepare operator and noise
	measure_config = task_config['measurement']
	operator_args = measure_config['operator']
	mask_gen = None

	if operator_args["name"] == "inpainting":
		mask_gen = mask_generator(**measure_config["mask_opt"])
		operator_args["mask_generator"] = mask_gen
		operator_args["image_size"] = measure_config["mask_opt"]["image_size"]  # Pass image_size explicitly
	else:
		mask_gen = None

	operator = get_operator(device=device, **operator_args)
	noiser = get_noise(**measure_config['noise'])

	# Prepare conditioning method
	cond_config = task_config['conditioning']
	cond_method = get_conditioning(cond_config['method'], operator, noiser, **cond_config['params'])
	measurement_cond_fn = cond_method.condition

	# Load sampler
	sampler_name = diffusion_config.get("sampler_name", "ddpm")
	sampler_args = {
		k: v for k, v in diffusion_config.items()
		if k not in {"betas", "rescale_timesteps", "sampler_name"}
	}
	sampler = create_sampler(sampler_name, diffusion_core=diffusion, **sampler_args)

	# Prepare output directories
	task_name = operator_args["name"]
	task_output_dir = os.path.join(save_dir, task_name)
	output_dirs = ["input", "label", "prog", "recon"]
	prepare_output_dirs(task_output_dir, output_dirs)

	# Use the "prog" directory directly for progress images
	progress_dir = os.path.join(task_output_dir, "prog")

	# Define sampling function
	sample_fn = partial(
		sampler.p_sample_loop,
		model=model,
		measurement_cond_fn=measurement_cond_fn,
		save_root=progress_dir  # Save intermediate images directly in "prog"
	)

	# Prepare dataset
	data_config = task_config['data']
	dataset = get_dataset(
		**data_config,
		transforms=transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		])
	)
	loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)

	# Perform inference
	for idx, ref_img in enumerate(tqdm(loader, desc="Inference")):
		ref_img = ref_img.to(device)
		fname = f"{str(idx).zfill(5)}.png"

		if operator_args["name"] == "inpainting":
			mask = mask_gen(ref_img).to(device)
			mask = mask[:, 0, :, :].unsqueeze(1)
			measurement_cond_fn = partial(cond_method.condition, mask=mask)
			sample_fn = partial(sample_fn, measurement_cond_fn=measurement_cond_fn)
			y = operator.forward(ref_img, mask=mask)
		else:
			y = operator.forward(ref_img)

		y_n = noiser(y)
		x_start = torch.randn_like(ref_img, device=device)

		# Perform sampling
		logger.info(f"Processing image {idx + 1}/{len(loader)}...")
		sample = sample_fn(x_start=x_start, measurement=y_n, record=True)

		# Save final results
		plt.imsave(os.path.join(task_output_dir, "input", fname), clear_color(y_n))
		plt.imsave(os.path.join(task_output_dir, "label", fname), clear_color(ref_img))
		plt.imsave(os.path.join(task_output_dir, "recon", fname), clear_color(sample))

		logger.info(f"Image {idx + 1} processing completed.")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', choices=['train', 'inference'], required=True)
	parser.add_argument('--model_checkpoint', type=str, help="Path to model checkpoint.")
	parser.add_argument('--configs', type=str, required=True)
	parser.add_argument('--task_config', type=str, required=False, default=None)
	parser.add_argument('--save_dir', type=str, default='./results')
	parser.add_argument('--epochs', type=int, help="Training epochs.")
	args = parser.parse_args()

	if args.mode == "train":
		if args.task_config:
			task_config = load_yaml(args.task_config)
		else:
			task_config = load_yaml(args.configs)  # Assume configs points directly to train_config.yaml

		model_config = load_yaml("configs/model_config.yaml")  # Point explicitly to model_config.yaml
		diffusion_config = load_yaml("configs/diffusion_config.yaml")  # Point explicitly to diffusion_config.yaml


		# Initialize diffusion
		valid_diffusion_args = {"betas", "rescale_timesteps"}
		diffusion_args = {k: v for k, v in diffusion_config.items() if k in valid_diffusion_args}

		# Provide a default linear beta schedule if "betas" is missing or incorrectly set
		if "betas" not in diffusion_args or diffusion_args["betas"] == "...":
			diffusion_args["betas"] = np.linspace(1e-4, 0.02, 1000)  # Default linear beta schedule

		diffusion = DiffusionCore(**diffusion_args)

		model = create_model(**model_config).to(device)
		optimizer = torch.optim.Adam(model.parameters(), lr=float(task_config["training"]["learning_rate"]))
		scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(task_config["training"]["lr_scheduler"]["step_size"]), gamma=float(task_config["training"]["lr_scheduler"]["gamma"]))
		criterion = nn.MSELoss()

		dataset = CustomImageDataset(root_dir=task_config["data"]["root_dir"], transform=transforms.Compose([
			transforms.RandomHorizontalFlip(),
			transforms.RandomResizedCrop(256, scale=task_config["augmentation"]["random_resized_crop"]["scale"]),
			transforms.ColorJitter(
				brightness=task_config["augmentation"]["color_jitter"]["brightness"],
				contrast=task_config["augmentation"]["color_jitter"]["contrast"],
				saturation=task_config["augmentation"]["color_jitter"]["saturation"],
				hue=task_config["augmentation"]["color_jitter"]["hue"]
			),
			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		]))

		val_split = 0.2
		val_size = int(len(dataset) * val_split)
		train_size = len(dataset) - val_size
		train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
		train_loader = DataLoader(train_dataset, batch_size=task_config["data"]["batch_size"], shuffle=True, num_workers=task_config["data"]["num_workers"])
		val_loader = DataLoader(val_dataset, batch_size=task_config["data"]["batch_size"], shuffle=False, num_workers=task_config["data"]["num_workers"])

		train_model(model, diffusion, train_loader, val_loader, args.epochs, optimizer, scheduler, criterion, args.save_dir)
	else:
		perform_inference(
			configs={
				'model_config': os.path.join(args.configs, "model_config.yaml"),
				'diffusion_config': os.path.join(args.configs, "diffusion_config.yaml"),
			},
			task_config=args.task_config,
			model_checkpoint=args.model_checkpoint,
			save_dir=args.save_dir
		)
