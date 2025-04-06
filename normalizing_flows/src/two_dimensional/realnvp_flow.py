from typing import Any, Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal


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
        nn.init.zeros_(self.final_layer.weight)
        nn.init.zeros_(self.final_layer.bias)

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
        
        # Final layer with zero-init
        return self.final_layer(h)


class Rescale(nn.Module):
    """Per-channel rescaling. Need a proper `nn.Module` so we can wrap it
    with `torch.nn.utils.weight_norm`.

    Args:
        num_channels (int): Number of channels in the input.
    """
    def __init__(self, num_channels: int):
        super(Rescale, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.weight * x
        return x


class RealNVPBijection(nn.Module):
    """RealNVP coupling layer for 2D data with checkerboard masking.
    """
    def __init__(self, in_channels: int, hidden_channels: int = 64, n_hidden_layers: int = 1, mask_reverse=False):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.n_hidden_layers = n_hidden_layers
        self.mask_reverse = mask_reverse
        
        # Neural network for computing scaling and translation parameters
        self.resnet = ResNet(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=in_channels * 2,  # scale and shift for each channel
            n_hidden_layers=n_hidden_layers
        )
        self.rescale = Rescale(in_channels)

    def _create_checkerboard_mask(self, height: int, width: int) -> torch.Tensor:
        # Create base 2x2 checkerboard pattern
        base_mask = torch.zeros(1, 1, 2, 2)
        base_mask[0, 0, 0, 0] = 1
        base_mask[0, 0, 1, 1] = 1
        if self.mask_reverse:
            base_mask = 1 - base_mask
        
        # Calculate number of repetitions needed
        h_repeats = height // 2
        w_repeats = width // 2
        
        # Repeat the pattern to match input dimensions
        mask = base_mask.repeat(1, 1, h_repeats, w_repeats)
        return mask
    
    def get_scale_and_shift(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Input shape: (batch_size, channels, height, width)
        mask = self._create_checkerboard_mask(x.shape[2], x.shape[3])
        params = self.resnet(x * mask)
        log_s, t = torch.chunk(params, 2, dim=1)  # chunk along channel dimension
        log_s = self.rescale(torch.tanh(log_s))
        # we want to apply the scale and shift only to the non-masked values
        log_s = log_s * (1 - mask)
        t = t * (1 - mask)
        return log_s, t

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        log_s_masked, t_masked = self.get_scale_and_shift(x)
        
        # Apply transformation
        z = x * torch.exp(log_s_masked) + t_masked
        
        # Compute log determinant
        log_det = log_s_masked.sum(dim=[1, 2, 3])
        
        return z, log_det

    @torch.no_grad()
    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        log_s_masked, t_masked = self.get_scale_and_shift(z)
        
        # Apply inverse transformation
        x = z * torch.exp(-log_s_masked) - t_masked
        
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