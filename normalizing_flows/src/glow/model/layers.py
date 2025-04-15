from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F


class ActNorm(nn.Module):

    def __init__(self, n_channels: int):
        super().__init__()
        self.n_channels = n_channels

        self.loc = nn.Parameter(torch.zeros(1, n_channels, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, n_channels, 1, 1))

        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))

    def initialize(self, x: torch.Tensor) -> None:
        # input shape: (batch_size, channels, height, width)
        with torch.no_grad():
            # Convert to (channels, batch_size, height, width) and flatten to (channels, batch_size * height * width)
            flattened = x.permute(1, 0, 2, 3).contiguous().view(self.n_channels, -1)
            # Calculate mean and std per channel
            loc = flattened.mean(dim=1).view(1, self.n_channels, 1, 1)
            scale = flattened.std(dim=1).view(1, self.n_channels, 1, 1)

            self.loc.data.copy_(loc)
            self.scale.data.copy_(1 / (scale + 1e-6))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, _, h, w = x.shape

        if self.initialized.item() == 0:
            self.initialize(x)
            self.initialized.fill_(1)

        log_abs = torch.log(torch.abs(self.scale))

        # Multiply by height and width to get logdet for each pixel
        logdet = h * w * torch.sum(log_abs)

        return self.scale * (x - self.loc), logdet

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        return z / self.scale + self.loc
