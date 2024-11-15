from PIL import Image
from enum import Enum
from typing import Union
from pathlib import Path
from torch import default_generator
from torch.utils.data import random_split, DataLoader
from matplotlib import pyplot as plt

from torchvision.datasets import VisionDataset
from torchvision.transforms import transforms, v2, InterpolationMode

# TODO: resize images based on min and max values instead of rescaling (by using ToTensor)
default_aug = [
    #transforms.Resize(),
    #transforms.RandomHorizontalFlip(p=0.5),
    #transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor()
]
block_inpaint = transforms.Compose([
    transforms.RandomErasing(),
    *default_aug
])
random_inpaint = transforms.Compose([*default_aug])

downscale_4x = transforms.Compose([
    v2.Resize(size=64, interpolation=InterpolationMode.BICUBIC),
    *default_aug
])
downscale_16x = transforms.Compose([
    v2.Resize(size=16, interpolation=InterpolationMode.BICUBIC),
    *default_aug
])

motion_blur = transforms.Compose([*default_aug])
gaussian_blur = transforms.Compose([
    v2.GaussianBlur(kernel_size=(61, 61), sigma=3.0),
    *default_aug
])
non_uniform_blur = transforms.Compose([*default_aug])

fourier_phase_transform = transforms.Compose([*default_aug])

transform_ops = [
    block_inpaint, random_inpaint, downscale_4x, downscale_16x,
    motion_blur, gaussian_blur, non_uniform_blur, fourier_phase_transform
]

tensor_to_pil = transforms.Compose([transforms.ToPILImage()])


class DatasetPaths(Enum):
    ffhq = Path('')  # TODO: add
    imagenet = Path('/home/space/datasets/imagenet/2012')


class MultiFolderDataset(VisionDataset):
    """Dataset consisting of multiple subfolders each containing training images"""
    def __init__(self, dataset_path: Union[DatasetPaths, Path], transform=None, file_search_extension='*.JPG'):
        self.path = dataset_path.value if type(dataset_path) is DatasetPaths else dataset_path
        self.images = [str(filename.absolute()) for filename in self.path.rglob(file_search_extension)]

        super().__init__(self.path, transform=transform)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        path = self.images[index]

        image = Image.open(path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        return image


def load_data(dataset='imagenet', transformation=None, n_train=0.8, n_test=0.2, **kwargs) -> tuple:
    try:
        dataset_path = DatasetPaths[dataset]
    except:
        dataset_path = DatasetPaths.imagenet

    ds = MultiFolderDataset(dataset_path=dataset_path, transform=transformation)
    if type(n_train) is not type(n_test):
        n_train = n_train if type(n_train) is int else len(ds) - n_test
        n_test = n_test if type(n_test) is int else len(ds) - n_train

    shuffle = kwargs.get('shuffle', True)
    batch_size = kwargs.get('batch_size', 32)
    split_generator = kwargs.get('split_generator', default_generator)
    train_ds, test_ds = random_split(ds, [n_train, n_test], split_generator)

    data_loader_args = dict(batch_size=batch_size, shuffle=shuffle)
    return DataLoader(train_ds, **data_loader_args), DataLoader(test_ds, **data_loader_args)


def show(x, outfile=None, transforms=None):
    x = tensor_to_pil(x)
    plt.imshow(x)

    if outfile:
        plt.savefig(outfile)
    else:
        plt.show()
