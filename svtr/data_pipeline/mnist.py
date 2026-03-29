from typing import Tuple, List

import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset

from svtr.constants import DATA_DIR


class ConcatenatedMNISTDataset(Dataset):
    """
    Usage example:
    ```
    from torch.utils.data import DataLoader

    train_dataset = ConcatenatedMNISTDataset(num_digits=(4, 6), train=True)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=32,
        shuffle=True
    )

    for images, targets, input_lengths, target_lengths in train_loader:
        pass
    ```
    """
    vocab = ['<BLK>', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
    vocab_size = len(vocab)
    indices = list(range(vocab_size))

    def __init__(self, num_digits: Tuple[int, int], train: bool = True, root: str = DATA_DIR):
        self.num_digits = num_digits
        self.train = train
        self.root = root

        # Create MNIST transformation sequence (images will have dynamic range [0, 1])
        mnist_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=[32, 32])
        ])
        # Load MNIST data
        self.mnist_dataset = torchvision.datasets.MNIST(
            root=root, train=train, download=True, transform=mnist_transform
        )
        # Character to index mapping
        self.char_to_idx = dict(zip(self.vocab, self.indices))
        self.idx_to_char = dict(zip(self.indices, self.vocab))

    def __len__(self) -> int:
        return int(len(self.mnist_dataset) / (sum(self.num_digits) / 2))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        """Generates concatenated mnist samples.
        Output shape of image: [1, H, W] where W is variable because the number of samples
        that will be concatenated is random.
        """
        # Determine the number of digits to concatenate
        np.random.seed(idx)
        num_digits = np.random.randint(self.num_digits[0], self.num_digits[1] + 1)
        indices = np.random.choice(len(self.mnist_dataset), size=num_digits)
        # Load the images/targets and concatenate them
        images = []
        targets = []
        for i in indices:
            image, char = self.mnist_dataset[i]
            images.append(image)
            targets.append(self.char_to_idx[str(char)])
        concatenated_image = torch.cat(images, dim=2)  # Concatenate along width dimension
        targets = torch.tensor(targets)

        img_w = concatenated_image.shape[2]
        input_length = int(np.ceil(img_w / 4))  # SVTR downsamples the width by a factor of 4
        target_length = targets.shape[0]

        return concatenated_image, targets, input_length, target_length

    @staticmethod
    def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, int, int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pads images and targets to the same size within the batch. Returns target lengths for CTC.

        Args:
            batch: List of (image [C, H, W], target [L], target_length).

        Returns:
            images: (B, C, H, W_max), targets: (B, L_max), input_lengths (B,) and target_lengths: (B,).
        """
        # Unpack the batch
        images, targets, input_lengths, target_lengths = zip(*batch)

        # Get batch size
        b = len(images)
        # Get image batch dimensions
        c = images[0].shape[0]
        max_h = max(img.shape[1] for img in images)  # Should be fixed height, but dynamically determining anyway
        max_w = max(img.shape[2] for img in images)
        # Get the length of the longest target
        max_l = max(tgt.shape[0] for tgt in targets)

        # Pad the images and targets so they can be stacked into a single tensor
        images_padded = torch.full((b, c, max_h, max_w), 1.0, dtype=images[0].dtype)
        targets_padded = torch.full((b, max_l), -1, dtype=targets[0].dtype)
        input_lengths = torch.tensor(input_lengths, dtype=torch.long)
        target_lengths = torch.tensor(target_lengths, dtype=torch.long)
        for i, (img, tgt, _, _) in enumerate(batch):
            _, h, w = img.shape
            images_padded[i, :, :h, :w] = img
            targets_padded[i, :tgt.shape[0]] = tgt

        return images_padded, targets_padded, input_lengths, target_lengths
