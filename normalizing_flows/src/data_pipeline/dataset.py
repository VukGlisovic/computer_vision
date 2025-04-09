from typing import List, Tuple, Optional
import random

import numpy as np
import torch
import torchvision
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
    
    # Create random subset dataset that samples 1000 random examples
    dataset = RandomSubsetDataset(base_dataset, subset_size=1000)
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn
    )

    # First iteration with first random subset
    for images, _ in dataloader:
        pass
        
    # Get new random subset
    dataset.new_random_subset()
    
    # Second iteration with new random subset
    for images, _ in dataloader:
        pass
    ```
    """
    def __init__(self, dataset: Dataset, subset_size: int, seed: Optional[int] = None):
        """
        Args:
            dataset: The base dataset to sample from
            subset_size: Number of samples to randomly select
            seed: Optional random seed for reproducibility
        """
        self.dataset = dataset
        self.subset_size = min(subset_size, len(dataset))
        self.seed = seed
        self._indices = None
        self.new_random_subset()
        
    def new_random_subset(self):
        """Generate a new random subset of indices."""
        if self.seed is not None:
            random.seed(self.seed)
        self._indices = random.sample(range(len(self.dataset)), self.subset_size)
        
    def __len__(self):
        return self.subset_size
        
    def __getitem__(self, idx):
        return self.dataset[self._indices[idx]]


class ChunkedDataset(Dataset):
    """
    A dataset wrapper that provides chunking of all the data such that you can evaluate
    more frequently.
    """
    def __init__(self, dataset: Dataset, n_chunks: int):
        """
        Args:
            dataset: The base dataset to sample from
            n_chunks: Number of chunks to split the dataset into
        """
        self.dataset = dataset
        self.n_chunks = n_chunks

        self.n = len(dataset)
        self.chunk_size = int(np.ceil(self.n / n_chunks))
        print(f"Number of samples in one chunk: {self.chunk_size}")
        self.chunk_nr = 0
        self._indices = []
        self._next_indices()
        
    def _next_indices(self):
        """Selects the next set of indices for the next chunk."""
        start_idx = self.chunk_nr * self.chunk_size
        end_idx = min((self.chunk_nr + 1) * self.chunk_size, self.n)
        self._indices = list(range(start_idx, end_idx))
        
    def advance_chunk(self):
        """Manually advance to the next chunk."""
        self.chunk_nr = (self.chunk_nr + 1) % self.n_chunks
        self._next_indices()
        
    def __len__(self):
        return len(self._indices)
        
    def __getitem__(self, idx):
        return self.dataset[self._indices[idx]]
