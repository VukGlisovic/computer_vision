from typing import Any

import torch
import torch.nn as nn


class ResNet(nn.Module):
    """Simple Residual CNN for generating scale and shift parameters in 2D.
    """
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int, n_hidden_layers: int = 1, act: Any = nn.ReLU):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.n_hidden_layers = n_hidden_layers
        self.act = act

        self.kernel_size = 3
        self.padding = 'same'
        
        # Initial projection if needed
        self.proj = None
        if in_channels != hidden_channels:
            self.proj = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)

        # Build main network
        self.residual_blocks = self.build_network()
        
        # Final layer with zero initialization
        self.final_layer = nn.Conv2d(hidden_channels, out_channels, kernel_size=self.kernel_size, padding=self.padding)

    def build_network(self):
        layers = []
        # Residual blocks
        for _ in range(self.n_hidden_layers):
            layers.append(
                nn.Sequential(
                    nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=self.kernel_size, padding=self.padding),
                    self.act(),
                    nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=self.kernel_size, padding=self.padding)
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
