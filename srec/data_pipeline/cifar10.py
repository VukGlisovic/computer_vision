"""CIFAR-10 dataset in the format expected by the SReC compressor."""

from typing import List

import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10


class SrecCifar10(CIFAR10):
    """CIFAR-10 images as RGB tensors of integer-valued floats in the 0-255 range.

    The SReC model models raw pixel symbols, so the images must keep their 0-255
    dynamic range instead of being scaled to 0-1. The labels are dropped because 
    the bits-per-sub-pixel objective is unsupervised.

    Args:
        root: Directory the dataset is stored in (and downloaded to).
        train: If True, use the training split; otherwise the test split.
        download: If True, download the dataset when it is not present in root.
        horizontal_flip: If True, randomly flip images left-right. Basically set
            this to True for training and False for testing.
    """

    def __init__(self,
                 root: str = "./data",
                 train: bool = True,
                 download: bool = True,
                 horizontal_flip: bool = False) -> None:
        transform: List = [transforms.RandomHorizontalFlip()] if horizontal_flip else []
        transform.append(transforms.PILToTensor())
        super().__init__(root=root, train=train, download=download, transform=transforms.Compose(transform))

    def __getitem__(self, idx: int) -> torch.Tensor:
        image, _ = super().__getitem__(idx)
        return image.float()
