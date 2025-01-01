import os
import torch
import argparse
import yaml
from functools import partial
from tqdm import tqdm
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
import matplotlib.pyplot as plt

from guided_diffusion.unet import create_model
from guided_diffusion.diffusion_core import DiffusionCore
from guided_diffusion.sampler_registry import create_sampler
from guided_diffusion.linear_operators import get_operator
from guided_diffusion.noise_registry import get_noise
from guided_diffusion.conditioning_registry import get_conditioning
from data.dataloader import get_dataset, get_dataloader
from utils.logger import get_logger
from utils.img_utils import clear_color, mask_generator

# Automatically select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_yaml(file_path: str) -> dict:
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def prepare_output_dirs(base_dir, sub_dirs):
    os.makedirs(base_dir, exist_ok=True)
    for sub_dir in sub_dirs:
        os.makedirs(os.path.join(base_dir, sub_dir), exist_ok=True)

def train_apply(configs, task_config, model_checkpoint, save_dir, apply_only=False):
    logger = get_logger()
    logger.info(f"Using device: {device}")

    # Load configurations
    model_config = load_yaml(configs['model_config'])
    diffusion_config = load_yaml(configs['diffusion_config'])
    task_config = load_yaml(task_config)

    # Initialize model
    model = create_model(**model_config).to(device)
    checkpoint = torch.load(model_checkpoint, map_location=device)
    model.load_state_dict(checkpoint.get("state_dict", checkpoint))
    model.eval()
    logger.info("Model loaded and set to evaluation mode.")

    # Prepare operator and noise
    measure_config = task_config['measurement']
    operator_name = measure_config['operator']['name']
    operator_args = measure_config['operator']
    mask_gen = None
    if 'mask_opt' in measure_config:  # For inpainting tasks
        mask_gen = mask_generator(**measure_config['mask_opt'])
        operator_args['mask_generator'] = mask_gen
    operator = get_operator(device=device, **operator_args)
    noiser = get_noise(**measure_config['noise'])

    # Prepare conditioning method
    cond_config = task_config['conditioning']
    cond_method = get_conditioning(cond_config['method'], operator, noiser, **cond_config['params'])
    measurement_cond_fn = cond_method.condition

    # Load sampler
    sampler = create_sampler(**diffusion_config)
    sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=measurement_cond_fn)

    # Prepare output directories
    output_dirs = ["input", "label", "recon", "progress"]
    out_path = os.path.join(save_dir, operator_name)
    prepare_output_dirs(out_path, output_dirs)

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
        logger.info(f"Processing image {fname}")

        if operator_name == 'inpainting':
            mask = mask_gen(ref_img).to(device)
            mask = mask[:, 0, :, :].unsqueeze(1)
            measurement_cond_fn = partial(cond_method.condition, mask=mask)
            sample_fn = partial(sample_fn, measurement_cond_fn=measurement_cond_fn)
            y = operator.forward(ref_img, mask=mask)
        else:
            y = operator.forward(ref_img)

        y_n = noiser(y)
        x_start = torch.randn_like(ref_img, device=device)
        sample = sample_fn(x_start=x_start, measurement=y_n, record=True, save_root=os.path.join(out_path, "progress"))

        # Save results
        plt.imsave(os.path.join(out_path, "input", fname), clear_color(y_n))
        plt.imsave(os.path.join(out_path, "label", fname), clear_color(ref_img))
        plt.imsave(os.path.join(out_path, "recon", fname), clear_color(sample))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_checkpoint', type=str, required=True, help="Path to trained model checkpoint.")
    parser.add_argument('--configs', type=str, required=True, help="Path to configuration YAML.")
    parser.add_argument('--task_config', type=str, required=True, help="Path to task-specific configuration YAML.")
    parser.add_argument('--save_dir', type=str, default='./results', help="Directory to save outputs.")
    parser.add_argument('--apply_only', action="store_true", help="Run in inference-only mode.")
    args = parser.parse_args()

    train_apply(
        configs={
            'model_config': os.path.join(args.configs, "model_config.yaml"),
            'diffusion_config': os.path.join(args.configs, "diffusion_config.yaml")
        },
        task_config=args.task_config,
        model_checkpoint=args.model_checkpoint,
        save_dir=args.save_dir,
        apply_only=args.apply_only
    )
