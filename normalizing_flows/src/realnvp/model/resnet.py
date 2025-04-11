from typing import Any

import torch
import torch.nn as nn

from normalizing_flows.src.realnvp.model.layer_utils import weight_norm


class ResNet(nn.Module):
    """Simple Residual CNN for generating scale and shift parameters in 2D.
    """
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int, n_residual_blocks: int = 1, in_factor: float = 1., act: Any = nn.ReLU):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.n_residual_blocks = n_residual_blocks
        self.in_factor = in_factor
        self.act = act

        self.kernel_size = 3
        self.padding = 'same'
        
        # Initial batchnorm and projection
        self.in_bn = nn.BatchNorm2d(in_channels)
        self.in_act = self.act()
        self.proj = weight_norm(nn.Conv2d(2 * in_channels, hidden_channels, kernel_size=1))

        # Build main network
        self.residual_blocks = self.build_network()
        
        # Final layer with zero initialization
        self.final_layer = nn.Sequential(
            nn.BatchNorm2d(self.hidden_channels),
            self.act(),
            weight_norm(nn.Conv2d(hidden_channels, out_channels, kernel_size=self.kernel_size, padding=self.padding))
        )

    def build_network(self):
        layers = []
        # Residual blocks
        for _ in range(self.n_residual_blocks):
            layers.append(
                nn.Sequential(
                    weight_norm(nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=self.kernel_size, padding=self.padding)),
                    nn.BatchNorm2d(self.hidden_channels),
                    self.act(),
                    weight_norm(nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=self.kernel_size, padding=self.padding)),
                    nn.BatchNorm2d(self.hidden_channels)
                )
            )
        return nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply layers to input before residual blocks
        x = self.in_bn(x)
        x = self.in_factor * x
        x = torch.cat([x, -x], dim=1)  # concat along channel dimension
        x = self.in_act(x)
        x = self.proj(x)
        
        # Residual blocks
        for rb in self.residual_blocks:
            x = x + rb(x)  # identity + residual block output
            x = self.act()(x)

        return self.final_layer(x)
