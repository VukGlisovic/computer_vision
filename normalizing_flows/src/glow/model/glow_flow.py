from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal

from normalizing_flows.src.realnvp.model.layers import PreprocessImages
from normalizing_flows.src.glow.model.blocks import SqueezeFlowStep


class Glow(nn.Module):

    def __init__(
        self, 
        in_channels: int,
        hidden_channels: int,
        size: int, 
        final_size: int, 
        n_flow_steps: int
    ):
        super().__init__()
        self.in_channels = in_channels
        self.size = size
        self.final_size = final_size
        self.n_flow_steps = n_flow_steps
        
        self.register_buffer('loc', torch.zeros((in_channels, size, size)))
        self.register_buffer('scale', torch.ones((in_channels, size, size)))
        
        # Create image preprocessing layer
        self.preprocess_images = PreprocessImages(alpha=0.05)

        self.squeeze_steps = nn.ModuleList()
        n_channels = in_channels
        while size > final_size:
            squeeze_step = SqueezeFlowStep(n_channels, hidden_channels, n_flow_steps)
            self.squeeze_steps.append(squeeze_step)
            size //= 2
            n_channels = squeeze_step.out_channels // 2  # half of latent continue, other half is split off

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
        for squeeze_step in self.squeeze_steps:
            z, log_det = squeeze_step(z)
            z, z_split = torch.chunk(z, 2, dim=1)
            total_log_det += log_det
            z_out.append(z_split)
        
        # Construct overall z
        for squeeze_step, z_split in reversed(list(zip(self.squeeze_steps, z_out))):
            z = torch.cat((z, z_split), dim=1)  # concat along channel dimension
            z = squeeze_step.squeeze.inverse(z)

        return z, total_log_det

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        z, total_log_det = self.forward(x)
        log_prob = self.base_dist.log_prob(z).sum(dim=[1, 2, 3]) + total_log_det
        return log_prob

    @torch.no_grad()
    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        # Extract individual z components
        z_out = []
        for squeeze_step in self.squeeze_steps:
            z = squeeze_step.squeeze(z)
            z, z_split = torch.chunk(z, 2, dim=1)
            z_out.append(z_split)

        # Run through all intermediate blocks (with upsampling)
        for squeeze_step, z_split in reversed(list(zip(self.squeeze_steps, z_out))):
            z = torch.cat((z, z_split), dim=1)
            z = squeeze_step.inverse(z)
        
        x = self.preprocess_images.inverse(z)
        return x

    @torch.no_grad()
    def sample(self, n_samples: int) -> torch.Tensor:
        z = self.base_dist.sample((n_samples,))
        x = self.inverse(z)
        return x
