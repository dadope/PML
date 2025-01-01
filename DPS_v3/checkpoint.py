import torch

# Load the saved checkpoint
checkpoint = torch.load("models/diffusion_ffhq_256_epoch10.pt", map_location="cpu")

# Check contents of the checkpoint
print(checkpoint.keys())  # Should include "state_dict" and possibly "optimizer"

# Inspect the model weights
state_dict = checkpoint["state_dict"]
for key, value in state_dict.items():
    print(f"{key}: {value.shape}")
