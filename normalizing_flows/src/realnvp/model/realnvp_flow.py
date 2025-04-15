from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal

from normalizing_flows.src.realnvp.model.layers import CheckerboardBijection2D, PreprocessImages
from normalizing_flows.src.realnvp.model.blocks import BlockBijection2D


class RealNVP(nn.Module):
    """
    RealNVP flow model for 2D data.
    """

    def __init__(
        self,
        in_channels: int,
        size: int = 32,  # Height and width; must be square input
        final_size: int = 4,
        n_cb_bijections: int = 3,
        n_cw_bijections: int = 3,
        n_final_bijections: int = 4,
        hidden_channels: int = 32,  # ResNet hidden channels (is doubled each block)
        n_residual_blocks: int = 1  # ResNet number of hidden layers
    ):
        super().__init__()
        self.in_channels = in_channels
        self.size = size
        self.final_size = final_size
        self.n_final_bijections = n_final_bijections

        self.register_buffer('loc', torch.zeros((in_channels, size, size)))
        self.register_buffer('scale', torch.ones((in_channels, size, size)))

        # Create image preprocessing layer
        self.preprocess_images = PreprocessImages(alpha=0.05)

        # Create intermediate blocks
        self.blocks = self.build_intermediate_blocks(hidden_channels, n_residual_blocks, n_cb_bijections, n_cw_bijections)

        # Create final block
        in_channels = self.blocks[-1].out_channels
        hidden_channels *= 2 ** len(self.blocks)
        self.final_layers = self.build_final_block(in_channels, hidden_channels, n_residual_blocks, n_bijections=n_final_bijections)

    def build_intermediate_blocks(self, hidden_channels: int, n_residual_blocks: int, n_cb_bijections: int, n_cw_bijections: int) -> nn.ModuleList:
        blocks = nn.ModuleList()
        in_channels = self.in_channels
        size = self.size
        while size > self.final_size:
            block = BlockBijection2D(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                n_cb_bijections=n_cb_bijections,
                n_cw_bijections=n_cw_bijections,
                n_residual_blocks=n_residual_blocks
            )
            in_channels = block.out_channels  # in_channels for next block
            hidden_channels *= 2
            size = size // 2
            blocks.append(block)
        return blocks
    
    def build_final_block(self, in_channels: int, hidden_channels: int, n_residual_blocks: int, n_bijections: int) -> nn.ModuleList:
        layers = nn.ModuleList()
        reverse_mask = False
        for _ in range(n_bijections):
            layers.append(CheckerboardBijection2D(in_channels, hidden_channels, n_residual_blocks, reverse_mask=reverse_mask))
            reverse_mask = not reverse_mask
        return layers

    @property
    def base_dist(self) -> Normal:
        return Normal(loc=self.loc, scale=self.scale)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
        z, total_log_det = self.forward(x)
        log_prob = self.base_dist.log_prob(z).sum(dim=[1, 2, 3]) + total_log_det
        return log_prob

    @torch.no_grad()
    def sample(self, n_samples: int) -> torch.Tensor:
        z = self.base_dist.sample((n_samples,))
        x = self.inverse(z)
        return x
