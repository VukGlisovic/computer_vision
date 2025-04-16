from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from normalizing_flows.src.realnvp.model.layers import Squeeze
from normalizing_flows.src.glow.model.layers import ActNorm, InvertibleConv2d, AffineCoupling


class FlowStep(nn.Module):
    """
    A single step in the Glow flow.
    """

    def __init__(self, in_channels: int, hidden_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.actnorm = ActNorm(in_channels)
        self.invconv = InvertibleConv2d(in_channels)
        self.coupling = AffineCoupling(in_channels, hidden_channels)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z, ldj1 = self.actnorm(x)
        z, ldj2 = self.invconv(z)
        z, ldj3 = self.coupling(z)
        logdet = ldj1 + ldj2 + ldj3
        return z, logdet

    @torch.no_grad()
    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        x = self.coupling.inverse(z)
        x = self.invconv.inverse(x)
        x = self.actnorm.inverse(x)
        return x


class SqueezeFlowStep(nn.Module):

    def __init__(self, in_channels: int, hidden_channels: int, n_flow_steps: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.n_flow_steps = n_flow_steps
        self.out_channels = in_channels * 4

        self.squeeze = Squeeze()
        self.flow_steps = nn.ModuleList(FlowStep(in_channels * 4, hidden_channels) for _ in range(n_flow_steps))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.squeeze(x)

        logdet = 0
        for flow_step in self.flow_steps:
            z, det = flow_step(z)
            logdet = logdet + det
        
        return z, logdet

    @torch.no_grad()
    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        for flow in reversed(self.flow_steps):
            z = flow.inverse(z)
        x = self.squeeze.inverse(z)
        return x
