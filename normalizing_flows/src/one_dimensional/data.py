from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn import datasets
import torch
from torch.utils.data import Dataset


def create_dataset(name: str, **kwargs) -> np.ndarray:
    """Choose a dataset name from https://scikit-learn.org/stable/api/sklearn.datasets.html

    Args:
        name: method names present on the sklearn.datasets module. E.g. 'make_moons' or 'make_circles'.
        kwargs: dataset specific keyword arguments. E.g. {'n_samples': 10000, 'noise': 0.05, 'random_state': 1}
            for the 'make_moons' dataset name.

    Returns:
        numpy array
    """
    dataset_fnc = getattr(datasets, name)
    features, _ = dataset_fnc(**kwargs)  # the labels we don't care about
    features = features.astype(np.float32)
    return features


def scatter_plot(features: np.ndarray, figsize: tuple = (8, 6), s: float = 2, alpha: float = 0.1) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(features[:, 0], features[:, 1], s=s, alpha=alpha)


def heatmap_plot(features: np.ndarray, figsize: tuple = (8, 6)) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    # Create a 2D histogram
    hist, xedges, yedges = np.histogram2d(features[:, 0], features[:, 1], bins=100)
    # Plots the 2D histogram as a heatmap
    ax.imshow(hist.T, extent=(xedges[0], xedges[-1], yedges[0], yedges[-1]), origin='lower', cmap=cm.viridis)


def density_heatmap_plot(model: Any, xs: torch.Tensor, ys: torch.Tensor, figsize: tuple = (10, 5)):
    xx, yy = torch.meshgrid(xs, ys, indexing='ij')
    points = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    device = next(model.parameters()).device
    points = points.to(device)

    with torch.no_grad():
        log_probs = model.log_prob(points)
    density = torch.exp(log_probs)
    density = density.reshape((len(xs), len(ys))).T

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        density.cpu().numpy(),
        extent=(min(xs), max(xs), min(ys), max(ys)),
        origin='lower',
        cmap='viridis'
    )
    plt.colorbar(im, label='Density')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Normalizing Flow Density Heatmap')


class OneDimensionalDataset(Dataset):
    """A simple dataset to loop over 2D data. Shape: [n_samples, n_features].
    """

    def __init__(self, features: np.ndarray):
        self.features = torch.from_numpy(features)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx: int):
        return self.features[idx]
