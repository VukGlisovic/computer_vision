from typing import Any, Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal


class Net(nn.Module):
    """Simple CNN for generating scale and shift parameters in 2D.
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
        self.net = self.build_network()

    def build_network(self):
        layers = [
            nn.Conv2d(self.in_channels, self.hidden_channels, kernel_size=self.kernel_size, padding=self.padding),
            self.act()
        ]
        for _ in range(self.n_hidden_layers):
            layers += [
                nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=self.kernel_size, padding=self.padding),
                self.act()
            ]
        layers.append(nn.Conv2d(self.hidden_channels, self.out_channels, kernel_size=self.kernel_size, padding=self.padding))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RealNVPBijection(nn.Module):
    """RealNVP coupling layer for 2D data with checkerboard masking.
    """
    def __init__(self, in_channels: int, hidden_channels: int = 64, n_hidden_layers: int = 1, height: int = 32, width: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.n_hidden_layers = n_hidden_layers
        
        # Create checkerboard mask with specified dimensions
        self.register_buffer('mask', self._create_checkerboard_mask(height, width))
        
        # Neural network for computing scaling and translation parameters
        self.net = Net(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=in_channels * 2,  # scale and shift for each channel
            n_hidden_layers=n_hidden_layers
        )

    def _create_checkerboard_mask(self, height: int, width: int) -> torch.Tensor:
        # Create base 2x2 checkerboard pattern
        base_mask = torch.zeros(1, 1, 2, 2)
        base_mask[0, 0, 0, 0] = 1
        base_mask[0, 0, 1, 1] = 1
        
        # Calculate number of repetitions needed
        h_repeats = height // 2
        w_repeats = width // 2
        
        # Repeat the pattern to match input dimensions
        mask = base_mask.repeat(1, 1, h_repeats, w_repeats)
        return mask

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Apply checkerboard mask
        x_masked = x * self.mask
        
        # Get scale and shift parameters
        params = self.net(x_masked)
        log_s, t = torch.chunk(params, 2, dim=1)
        
        # Apply transformation
        z = x_masked + (1 - self.mask) * (x * torch.exp(log_s) + t)
        
        # Compute log determinant
        log_det = ((1 - self.mask) * log_s).sum(dim=[1, 2, 3])
        
        return z, log_det

    @torch.no_grad()
    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        # Apply checkerboard mask
        x = z * self.mask
        
        # Get scale and shift parameters
        params = self.net(x)
        log_s, t = torch.chunk(params, 2, dim=1)
        
        # Apply inverse transformation
        x = x * self.mask + (1 - self.mask) * ((x - t) * torch.exp(-log_s))
        
        return x


class RealNVPFlow(nn.Module):
    """RealNVP flow model for 2D data.
    """
    def __init__(
        self,
        in_channels: int,
        height: int = 32,
        width: int = 32,
        hidden_channels: int = 64,
        n_hidden_layers: int = 1,
        n_coupling_layers: int = 4
    ):
        super().__init__()
        self.in_channels = in_channels
        self.height = height
        self.width = width
        self.hidden_channels = hidden_channels
        self.n_hidden_layers = n_hidden_layers
        self.n_coupling_layers = n_coupling_layers
        
        # Initialize base distribution parameters
        self.register_buffer('loc', torch.zeros(in_channels, height, width))
        self.register_buffer('scale', torch.ones(in_channels, height, width))
        
        # Build the flow
        self.layers = self.build_model()

    def build_model(self) -> nn.ModuleList:
        layers = []
        for _ in range(self.n_coupling_layers):
            layers.append(RealNVPBijection(
                in_channels=self.in_channels,
                hidden_channels=self.hidden_channels,
                n_hidden_layers=self.n_hidden_layers,
                height=self.height,
                width=self.width
            ))
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
        z = x
        total_log_det = torch.zeros(x.shape[0], device=x.device)

        for layer in self.layers:
            z, log_det = layer(z)
            total_log_det = total_log_det + log_det

        return z, total_log_det

    @torch.no_grad()
    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Inverse pass through all coupling layers.

        Args:
            z (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)

        Returns:
            torch.Tensor: Reconstructed input
        """
        x = z
        for layer in reversed(self.layers):
            x = layer.inverse(x)
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
        for layer in reversed(self.layers):
            z = layer.inverse(z)
        return z 