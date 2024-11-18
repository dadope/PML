import torch
from torchvision.transforms import transforms, v2, InterpolationMode

from bin.motion_blur import MotionBlur

# TODO: resize images based on min and max values instead of rescaling (by using ToTensor)
DEFAULT_AUGMENTATIONS = [
    transforms.Resize((256, 256)),
    #transforms.RandomHorizontalFlip(p=0.5),
    #transforms.RandomVerticalFlip(p=0.5),
    #transforms.CenterCrop((256, 256)),
    v2.GaussianBlur(kernel_size=(61, 61), sigma=0.005),
    transforms.ToTensor()
]

non_uniform_transform = lambda image: torch.log(image + 0.00001)
fourier_magnitute = lambda image: torch.abs(torch.fft.fftn(image))
create_transform = lambda transform: transforms.Compose([transform, *DEFAULT_AUGMENTATIONS])

class RandomInpaint(v2.Transform):
    def __init__(self, p=0.9):
        super().__init__()
        self.p = p

    def forward(self, img):
        mask = torch.rand(1, img.size(1), img.size(2)) > self.p
        return img * mask



block_inpaint = transforms.Compose([*DEFAULT_AUGMENTATIONS, transforms.RandomErasing(p=1.0)])
downscale_4x = create_transform(v2.Resize(size=64, interpolation=InterpolationMode.BICUBIC))
downscale_16x = create_transform(v2.Resize(size=16, interpolation=InterpolationMode.BICUBIC))
gaussian_blur = create_transform(v2.GaussianBlur(kernel_size=(61, 61), sigma=3.0))
motion_blur = create_transform(MotionBlur())
random_inpaint = transforms.Compose([*DEFAULT_AUGMENTATIONS, RandomInpaint()])
fourier_phase_transform = transforms.Compose([*DEFAULT_AUGMENTATIONS, fourier_magnitute])
non_uniform = transforms.Compose([*DEFAULT_AUGMENTATIONS, non_uniform_transform])

transform_ops = {
    "Gaussian blur": gaussian_blur,
    "Motion blur": motion_blur,
    "Downscaling 4x": downscale_4x,
    "Downscaling 16x": downscale_16x,
    "Random inpainting": random_inpaint,
    "Block inpainting": block_inpaint,
    "Non Uniform transform": non_uniform,
    "Fourier Phase Transform": fourier_phase_transform
}

tensor_to_pil = transforms.Compose([transforms.ToPILImage()])
