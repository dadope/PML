import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
from PIL import Image
import time
import matplotlib.pyplot as plt
from guided_diffusion.unet import create_model
from guided_diffusion.diffusion_core import DiffusionCore
from guided_diffusion.sampler_registry import create_sampler

# Automatically select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Custom Dataset for Unlabeled Images
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

# Dataset preparation
data_dir = "./data/ffhq256/"
transform = transforms.Compose([
	transforms.RandomHorizontalFlip(),
	transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
	transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
	transforms.ToTensor(),
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = CustomImageDataset(root_dir=data_dir, transform=transform)

# Split dataset into training and validation sets
val_split = 0.1
val_size = int(len(dataset) * val_split)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)

# Model and diffusion configuration
model = create_model(
	image_size=256,
	num_channels=256,  # Increased capacity
	num_res_blocks=3,
	channel_mult="1,2,4,8",
	class_cond=False,
	use_fp16=False,
	dropout=0.1
).to(device)

diffusion = DiffusionCore(
	betas=torch.linspace(1e-4, 0.02, 1000),
	rescale_timesteps=False
)

sampler = create_sampler(
	sampler_name="ddpm",
	steps=1000,
	noise_schedule="linear",
	model_mean_type="epsilon",
	model_var_type="learned_range"
)

# Optimizer, loss, and learning rate scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
criterion = nn.MSELoss()

# Training loop
epochs = 50
loss_history = []
val_loss_history = []

for epoch in range(epochs):
	start_time = time.time()
	model.train()
	epoch_loss = 0

	for i, inputs in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}")):
		inputs = inputs.to(device).requires_grad_()  # Ensure inputs require gradients
		optimizer.zero_grad()

		# Diffusion forward pass
		t = torch.randint(0, 1000, (inputs.size(0),), device=device)
		noise = torch.randn_like(inputs, requires_grad=True).to(device)  # Ensure noise requires gradients
		noisy_inputs = diffusion.q_sample(inputs, t, noise=noise).requires_grad_()

		# Model prediction and loss
		outputs = model(noisy_inputs, t)
		assert outputs.requires_grad, "Model outputs are not connected to the graph."

		loss = criterion(outputs, noise)
		assert loss.requires_grad, "Loss is not connected to the computational graph."

		# Backward and optimization
		loss.backward()
		optimizer.step()
		epoch_loss += loss.item()


	scheduler.step()
	avg_epoch_loss = epoch_loss / len(train_loader)
	loss_history.append(avg_epoch_loss)

	# Validation loop
	model.eval()
	val_loss = 0
	with torch.no_grad():
		for inputs in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{epochs}"):
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
	remaining_time = epoch_duration * (epochs - epoch - 1)

	print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Time: {epoch_duration:.2f}s, Estimated Remaining: {remaining_time / 60:.2f}min")

	# Save checkpoint
	os.makedirs("./models", exist_ok=True)
	torch.save({
		"epoch": epoch,
		"state_dict": model.state_dict(),
		"optimizer": optimizer.state_dict()
	}, f"./models/diffusion_ffhq_256_epoch{epoch+1}.pt")

	# Save sample outputs every 5 epochs
	if (epoch + 1) % 5 == 0:
		sample_inputs = next(iter(val_loader)).to(device)[:8]
		t_sample = torch.randint(0, 1000, (sample_inputs.size(0),), device=device)
		noisy_sample = diffusion.q_sample(sample_inputs, t_sample, noise=torch.randn_like(sample_inputs).to(device))
		sample_outputs = model(noisy_sample, t_sample)
		os.makedirs("./outputs", exist_ok=True)
		save_image((sample_outputs + 1) / 2, f"./outputs/epoch_{epoch+1}.png", nrow=4)

# Plot training and validation loss
plt.plot(loss_history, label="Training Loss")
plt.plot(val_loss_history, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Validation Loss")
plt.savefig("./loss_plot.png")
plt.show()
