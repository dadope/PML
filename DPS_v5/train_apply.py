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
		"""
		Args:
			root_dir (str): Directory with all the images.
			transform (callable, optional): Transform to apply to the images.
		"""
		self.root_dir = root_dir
		self.file_names = [f for f in os.listdir(root_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
		self.transform = transform

	def __len__(self):
		return len(self.file_names)

	def __getitem__(self, idx):
		img_path = os.path.join(self.root_dir, self.file_names[idx])
		image = Image.open(img_path).convert("RGB")  # Convert to RGB
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
	loss_history = []
	val_loss_history = []

	for epoch in range(epochs):
		start_time = time.time()
		model.train()
		epoch_loss = 0

		for i, inputs in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}")):
			inputs = inputs.to(device).requires_grad_()
			optimizer.zero_grad()

			# Diffusion forward pass
			t = torch.randint(0, 1000, (inputs.size(0),), device=device)
			noise = torch.randn_like(inputs, requires_grad=True).to(device)
			noisy_inputs = diffusion.q_sample(inputs, t, noise=noise).requires_grad_()

			# Model prediction and loss
			outputs = model(noisy_inputs, t)
			loss = criterion(outputs, noise)

			# Backward and optimization
			loss.backward()
			optimizer.step()
			epoch_loss += loss.item()

		scheduler.step()
		avg_epoch_loss = epoch_loss / len(train_loader)
		loss_history.append(avg_epoch_loss)

		# Validation loop
		if len(val_loader) == 0:
			logger.warning("Validation loader is empty. Skipping validation for this epoch.")
			continue

		model.eval()
		val_loss = 0
		with torch.no_grad():
			for inputs in val_loader:
				inputs = inputs.to(device)
				t = torch.randint(0, 1000, (inputs.size(0),), device=device)
				noise = torch.randn_like(inputs).to(device)
				noisy_inputs = diffusion.q_sample(inputs, t, noise=noise)

				outputs = model(noisy_inputs, t)
				val_loss += criterion(outputs, noise).item()

		avg_val_loss = val_loss / len(val_loader)
		val_loss_history.append(avg_val_loss)

		# Time and logs
		end_time = time.time()
		epoch_duration = end_time - start_time
		logger.info(
			f"Epoch {epoch + 1}/{epochs}, Loss: {avg_epoch_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Time: {epoch_duration:.2f}s"
		)

		# Save checkpoint
		os.makedirs(save_dir, exist_ok=True)
		torch.save(
			{
				"epoch": epoch,
				"state_dict": model.state_dict(),
				"optimizer": optimizer.state_dict(),
			},
			f"{save_dir}/model_epoch_{epoch + 1}.pt",
		)

	# Plot training and validation loss
	plt.plot(loss_history, label="Training Loss")
	plt.plot(val_loss_history, label="Validation Loss")
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.legend()
	plt.title("Training and Validation Loss")
	plt.savefig(f"{save_dir}/loss_plot.png")
	plt.show()
 
def perform_inference(configs, task_config, model_checkpoint, save_dir):
    logger = get_logger()
    logger.info("Starting inference...")

    # Load configurations
    model_config = load_yaml(configs['model_config'])
    diffusion_config = load_yaml(configs['diffusion_config'])
    task_config = load_yaml(task_config)

    # Initialize model
    model = create_model(**model_config).to(device)
    checkpoint = torch.load(model_checkpoint, map_location=device)
    model.load_state_dict(checkpoint.get("state_dict", checkpoint))
    model.eval()

    # Prepare operator and noise
    measure_config = task_config['measurement']
    operator_args = measure_config['operator']
    mask_gen = None

    # Handle inpainting-specific masking
    if operator_args["name"] == "inpainting":
        mask_gen = mask_generator(**measure_config["mask_opt"])  # Generate mask based on task configuration
        operator_args["mask_generator"] = mask_gen

    operator = get_operator(device=device, **operator_args)
    noiser = get_noise(**measure_config['noise'])

    # Prepare conditioning method
    cond_config = task_config['conditioning']
    cond_method = get_conditioning(cond_config['method'], operator, noiser, **cond_config['params'])
    measurement_cond_fn = cond_method.condition

    # Load sampler
    sampler_name = diffusion_config.get("sampler_name", "ddpm")
    sampler_args = {k: v for k, v in diffusion_config.items() if k not in {"betas", "rescale_timesteps"}}
    sampler_args.pop("sampler_name", None)  # Ensure 'sampler_name' is not passed twice
    sampler = create_sampler(sampler_name, **sampler_args)
    
    # Set sample function with correct save_root
    task_name = operator_args["name"]
    task_output_dir = os.path.join(save_dir, task_name)  # e.g., results/inpainting
    output_dirs = ["input", "label", "prog", "recon"]
    prepare_output_dirs(task_output_dir, output_dirs)

    sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=measurement_cond_fn, save_root=os.path.join(task_output_dir, "prog"))

    # Prepare dataset
    data_config = task_config['data']
    dataset = get_dataset(**data_config, transforms=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]))
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
        sample = sample_fn(x_start=x_start, measurement=y_n, record=True)  # Save_root is handled correctly now

        # Save results in appropriate folders
        plt.imsave(os.path.join(task_output_dir, "input", fname), clear_color(y_n))
        plt.imsave(os.path.join(task_output_dir, "label", fname), clear_color(ref_img))
        plt.imsave(os.path.join(task_output_dir, "recon", fname), clear_color(sample))


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', choices=['train', 'inference'], required=True, help="Run mode: 'train' or 'inference'.")
	parser.add_argument('--model_checkpoint', type=str, help="Path to the model checkpoint.")
	parser.add_argument('--configs', type=str, required=True, help="Path to configuration YAML.")
	parser.add_argument('--task_config', type=str, required=True, help="Path to task-specific configuration YAML.")
	parser.add_argument('--save_dir', type=str, default='./results', help="Directory to save outputs.")
	parser.add_argument('--epochs', type=int, help="Number of training epochs (only for training).")
	args = parser.parse_args()

	if args.mode == "train":
		# Training setup
		task_config = load_yaml(args.task_config)
		model_config = load_yaml(os.path.join(args.configs, "model_config.yaml"))
		diffusion_config = load_yaml(os.path.join(args.configs, "diffusion_config.yaml"))
		valid_diffusion_args = {"betas", "rescale_timesteps"}
		diffusion_args = {k: v for k, v in diffusion_config.items() if k in valid_diffusion_args}
		if "betas" not in diffusion_args or diffusion_args["betas"] == "...":
			diffusion_args["betas"] = np.linspace(1e-4, 0.02, 1000)  # Default linear beta schedule
		diffusion = DiffusionCore(**diffusion_args)

		model = create_model(**model_config).to(device)
		learning_rate = float(task_config["training"]["learning_rate"])  # Explicit conversion
		step_size = int(task_config["training"]["lr_scheduler"]["step_size"])  # Explicit conversion
		gamma = float(task_config["training"]["lr_scheduler"]["gamma"])  # Explicit conversion

		optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
		scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

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
