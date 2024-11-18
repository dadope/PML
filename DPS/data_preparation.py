from PIL import Image
from enum import Enum
from typing import Union
from pathlib import Path
from matplotlib import pyplot as plt

from torch import default_generator
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import VisionDataset

from DPS.image_transforms import tensor_to_pil

FILE_SEARCH_EXTENSION = ["*.png", "*.jpg", "*.jpeg", "*.JPG", "*.JPEG"]


class DatasetPaths(Enum):
    ffhq = Path('/home/space/datasets/ffhq')
    imagenet = Path('/home/space/datasets/imagenet/2012')


class MultiFolderDataset(VisionDataset):
    """Dataset consisting of multiple subfolders each containing training images"""

    def __init__(self, dataset_path: Union[DatasetPaths, Path], transform=None):
        self.path = getattr(dataset_path, "value", dataset_path)

        self.images = [str(fname.absolute()) for ext in FILE_SEARCH_EXTENSION for fname in self.path.rglob(ext)]
        super().__init__(self.path, transform=transform)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]

        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        return image


def get_datasets(dataset='imagenet', transformation=None, n_train=0.8, n_test=0.2, **kwargs) -> tuple:
    dataset_path = DatasetPaths[dataset] if type(dataset) is str else dataset

    ds = MultiFolderDataset(dataset_path=dataset_path, transform=transformation)
    if type(n_train) is not type(n_test):
        n_train = n_train if type(n_train) is int else len(ds) - n_test
        n_test = n_test if type(n_test) is int else len(ds) - n_train

    split_generator = kwargs.get('split_generator', default_generator)
    train_ds, test_ds = random_split(ds, [n_train, n_test], split_generator)

    return train_ds, test_ds


def load_data(**kwargs) -> tuple:
    shuffle = kwargs.get('shuffle', True)
    batch_size = kwargs.get('batch_size', 32)
    data_loader_args = dict(batch_size=batch_size, shuffle=shuffle)

    train_ds, test_ds = get_datasets(**kwargs)
    return DataLoader(train_ds, **data_loader_args), DataLoader(test_ds, **data_loader_args)


def show(x, outfile=None):
    x = tensor_to_pil(x)
    plt.imshow(x)

    if outfile:
        plt.savefig(outfile)
    else:
        plt.show()
