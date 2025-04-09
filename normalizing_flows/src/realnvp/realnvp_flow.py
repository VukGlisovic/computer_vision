from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal

from normalizing_flows.src.realnvp.layers import CouplingBijection2D, PreprocessImages
from normalizing_flows.src.realnvp.blocks import BlockBijection2D


class RealNVP(nn.Module):
    """RealNVP flow model for 2D data.
    """
    def __init__(
        self,
        in_channels: int,
        size: int = 32,  # Height and width; must be square input
        hidden_channels: int = 32,  # is doubled each block
        n_hidden_layers: int = 1,
        final_size: int = 4
    ):
        super().__init__()
        self.in_channels = in_channels
        self.size = size
        self.n_hidden_layers = n_hidden_layers
        self.final_size = final_size

        self.loc = nn.Parameter(torch.zeros((in_channels, size, size)), requires_grad=False)
        self.scale = nn.Parameter(torch.ones((in_channels, size, size)), requires_grad=False)

        # Create image preprocessing layer
        self.preprocess_images = PreprocessImages(alpha=0.05)

        # Create intermediate blocks
        self.blocks = self.build_intermediate_blocks(hidden_channels)

        # Create final block
        in_channels = self.blocks[-1].out_channels
        hidden_channels *= 2 ** len(self.blocks)
        self.final_layers = self.build_final_block(in_channels, hidden_channels, n_bijections=4)

    def build_intermediate_blocks(self, hidden_channels: int) -> nn.ModuleList:
        blocks = []
        in_channels = self.in_channels
        size = self.size
        while size > self.final_size:
            block = BlockBijection2D(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                n_hidden_layers=1
            )
            in_channels = block.out_channels  # in_channels for next block
            hidden_channels *= 2
            size = size // 2
            blocks.append(block)
        return nn.ModuleList(blocks)
    
    def build_final_block(self, in_channels: int, hidden_channels: int, n_bijections: int) -> nn.Module:
        layers = []
        reverse_mask = False
        for _ in range(n_bijections):
            layers.append(CouplingBijection2D(in_channels, hidden_channels, self.n_hidden_layers, 'checkerboard', reverse_mask=reverse_mask))
            reverse_mask = not reverse_mask
        return nn.ModuleList(layers)

    @property
    def base_dist(self) -> Normal:
        return Normal(loc=self.loc, scale=self.scale)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through all coupling layers.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (transformed_x, total_log_det_jacobian)
        """
        total_log_det = torch.zeros(x.shape[0], device=x.device)

        # Preprocess images
        z, log_det = self.preprocess_images(x)
        total_log_det += log_det

        # Run through all intermediate blocks (with downsampling)
        z_out = []
        for block in self.blocks:
            z, log_det = block(z)
            z, z_split = torch.chunk(z, 2, dim=1)
            total_log_det += log_det
            z_out.append(z_split)
        
        # Run through final block (no downsampling)
        for layer in self.final_layers:
            z, log_det = layer(z)
            total_log_det += log_det

        # Construct overall z
        for block, z_split in reversed(list(zip(self.blocks, z_out))):
            z = torch.cat((z, z_split), dim=1)  # concat along channel dimension
            z = block.squeeze_permute.inverse(z)

        return z, total_log_det

    @torch.no_grad()
    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Inverse pass through all coupling layers.

        Args:
            z (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)

        Returns:
            torch.Tensor: Reconstructed input
        """
        # Extract individual z components
        z_out = []
        for block in self.blocks:
            z = block.squeeze_permute(z)
            z, z_split = torch.chunk(z, 2, dim=1)
            z_out.append(z_split)
        
        # Run through final block (no downsampling)
        for layer in reversed(self.final_layers):
            z = layer.inverse(z)

        # Run through all intermediate blocks (with downsampling)
        for block, z_split in reversed(list(zip(self.blocks, z_out))):
            z = torch.cat((z, z_split), dim=1)
            z = block.inverse(z)
        
        x = self.preprocess_images.inverse(z)
        return x

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Compute log probability of input samples.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)

        Returns:
            torch.Tensor: Log probabilities
        """
        z, total_log_det = self.forward(x)
        log_prob = self.base_dist.log_prob(z).sum(dim=[1, 2, 3]) + total_log_det
        return log_prob

    @torch.no_grad()
    def sample(self, n_samples: int) -> torch.Tensor:
        """Generate samples from the model.

        Args:
            n_samples (int): Number of samples to generate

        Returns:
            torch.Tensor: Generated samples
        """
        z = self.base_dist.sample((n_samples,))
        x = self.inverse(z)
        return x
