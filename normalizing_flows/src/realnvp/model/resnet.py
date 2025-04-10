from typing import Any

import torch
import torch.nn as nn


class ResNet(nn.Module):
    """Simple Residual CNN for generating scale and shift parameters in 2D.
    """
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int, n_residual_blocks: int = 1, act: Any = nn.ReLU):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.n_residual_blocks = n_residual_blocks
        self.act = act

        self.kernel_size = 3
        self.padding = 'same'
        
        # Initial projection if needed
        self.proj = None
        if in_channels != hidden_channels:
            self.proj = weight_norm(nn.Conv2d(in_channels, hidden_channels, kernel_size=1))

        # Build main network
        self.residual_blocks = self.build_network()
        
        # Final layer with zero initialization
        self.final_layer = weight_norm(nn.Conv2d(hidden_channels, out_channels, kernel_size=self.kernel_size, padding=self.padding))

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
        # Initial projection if needed
        h = self.proj(x) if self.proj is not None else x
        
        # Residual blocks
        for rb in self.residual_blocks:
            identity = h
            h = rb(h)
            h = h + identity
            h = self.act()(h)

        return self.final_layer(h)


def weight_norm(layer: Any) -> Any:
    """The reason this method exists, is for documentation. The idea is taken from:

    "Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks" - https://arxiv.org/abs/1602.07868

    The idea is that it stabilizes training by decomposing the weights into a
    direction vector and a magnitude scalar. These individual components are then
    updated through backpropagation.
    """
    return nn.utils.parametrizations.weight_norm(layer)
