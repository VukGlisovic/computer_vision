import os

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset

from svtr.constants import PROJECT_PATH


class ConcatenatedMNISTDataset(Dataset):
    """
    Usage example:
    ```
    from torch.utils.data import DataLoader

    train_dataset = ConcatenatedMNISTDataset(num_digits=5, train=True)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=32,
        shuffle=True
    )

    for images, labels in train_loader:
        pass
    ```
    """

    def __init__(self, num_digits, train=True, root=os.path.join(PROJECT_PATH, 'data')):
        self.num_digits = num_digits
        self.train = train
        self.root = root
        # create MNIST transformation sequence (images will have dynamic range [0, 1])
        mnist_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=[32, 32])
        ])
        # load MNIST data
        self.mnist_dataset = torchvision.datasets.MNIST(
            root=root, train=train, download=True, transform=mnist_transform
        )

    def __len__(self):
        return len(self.mnist_dataset) // self.num_digits

    def __getitem__(self, index):
        """
        Output shape of image: [C, H, W]
        Since the targets are all digits and already nicely aligned with their indices,
        we don't have to apply a character to index mapping.
        """
        images = []
        targets = []
        start = index * self.num_digits
        for i in range(start, start +self.num_digits):
            image, target = self.mnist_dataset[i]
            images.append(image)
            targets.append(target)
        concatenated_image = torch.cat(images, dim=2)  # Concatenate along width dimension
        targets = torch.tensor(targets)
        return concatenated_image, targets
