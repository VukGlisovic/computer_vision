from typing import List, Tuple, Optional
import random

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset


class CelebADataset(torchvision.datasets.CelebA):
    """Dataset class for CelebA images with error handling in __getitem__.
    
    Usage example:
    ```
    from torch.utils.data import DataLoader
    from normalizing_flows.data_pipeline.collate import collate_fn

    dataset = CelebADataset(root='./data', split='train')
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn
    )

    for images, _ in dataloader:
        pass
    ```
    """
    def __init__(self, root='./data', split='train', download=True, transform=None):
        super().__init__(
            root=root,
            split=split,
            download=download,
            transform=transform
        )

    def __getitem__(self, index):
        """Returns (image, None) for compatibility with collate_fn.
        Returns None if loading fails.
        """
        try:
            image, labels = super().__getitem__(index)
            return image, labels
        except Exception as e:
            print(f"Error loading sample at index {index}: {str(e)}")
            return None 
    
    @staticmethod
    def collate_fn_skip_errors(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Custom collate function that filters out failed samples and creates a batch with the remaining valid samples.
        
        Args:
            batch: List of tuples containing (image, target) pairs. Some pairs may be None if loading failed.
            
        Returns:
            Tuple of batched images and targets, with failed samples removed.
        """
        # Filter out None values (failed samples)
        valid_batch = [item for item in batch if item is not None]
        
        if not valid_batch:
            # If all samples failed, return empty tensors
            return torch.empty(0), torch.empty(0)
        
        # Unzip the valid samples
        images, targets = zip(*valid_batch)
        
        # Stack the images and targets
        images = torch.stack(images)
        targets = torch.stack(targets)
        
        return images, targets 


class RandomSubsetDataset(Dataset):
    """A dataset wrapper that provides random sampling of a subset of the original dataset.
    
    Usage example:
    ```
    from torch.utils.data import DataLoader
    from normalizing_flows.data_pipeline.collate import collate_fn

    # Create base dataset
    base_dataset = CelebADataset(root='./data', split='train')
    
    # Create random subset dataset that samples 1000 random examples per epoch
    dataset = RandomSubsetDataset(base_dataset, subset_size=1000)
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn
    )

    for images, _ in dataloader:
        pass
    ```
    """
    def __init__(self, dataset: Dataset, subset_size: int, seed: Optional[int] = None):
        """
        Args:
            dataset: The base dataset to sample from
            subset_size: Number of samples to randomly select per epoch
            seed: Optional random seed for reproducibility
        """
        self.dataset = dataset
        self.subset_size = min(subset_size, len(dataset))
        self.seed = seed
        self._indices = None
        self._reset_indices()
        
    def _reset_indices(self):
        """Resets the random indices for the next epoch."""
        if self.seed is not None:
            random.seed(self.seed)
        self._indices = random.sample(range(len(self.dataset)), self.subset_size)
        
    def __len__(self):
        return self.subset_size
        
    def __getitem__(self, idx):
        if idx == 0:
            self._reset_indices()  # Reset indices at the start of each epoch
        return self.dataset[self._indices[idx]] 