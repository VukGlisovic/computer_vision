from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal

from normalizing_flows.src.realnvp.resnet import ResNet


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


class Squeeze(nn.Module):
    """Squeezes a C x H x W tensor into a 4C x H/2 x W/2 tensor.
    """
    def __init__(self):
        super(Squeeze, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, c, h, w = x.shape
        x = x.reshape(bs, c, h//2, 2, w//2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4)
        x = x.reshape(bs, c*4, h//2, w//2)
        return x
    
    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        bs, c, h, w = x.shape
        x = x.reshape(bs, c//4, 2, 2, h, w)
        x = x.permute(0, 1, 4, 2, 5, 3)
        x = x.reshape(bs, c//4, h*2, w*2)
        return x


class SqueezePermute(nn.Module):
    """Squeezes a C x H x W tensor into a 4C x H/2 x W/2 tensor and permutes the channels.
    """
    def __init__(self, in_channels: int):
        super(SqueezePermute, self).__init__()
        self.in_channels = in_channels
        self.perm_weight = nn.Parameter(self._create_perm_weight(), requires_grad=False)
    
    def _create_perm_weight(self) -> torch.Tensor:
        c = self.in_channels
        # Defines permutation of input channels (shape is (4, 1, 2, 2)).
        squeeze_matrix = torch.tensor(
            [[[[1., 0.], [0., 0.]]],
            [[[0., 0.], [0., 1.]]],
            [[[0., 1.], [0., 0.]]],
            [[[0., 0.], [1., 0.]]]]
        )
        perm_weight = torch.zeros((4 * c, c, 2, 2))
        for c_idx in range(c):
            perm_weight[c_idx * 4: (c_idx + 1) * 4, c_idx: c_idx + 1, :, :] = squeeze_matrix
        shuffle_channels = torch.tensor(
            [c_idx * 4 for c_idx in range(c)]
            + [c_idx * 4 + 1 for c_idx in range(c)]
            + [c_idx * 4 + 2 for c_idx in range(c)]
            + [c_idx * 4 + 3 for c_idx in range(c)]
        )
        perm_weight = perm_weight[shuffle_channels, :, :, :]
        return perm_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.conv2d(x, self.perm_weight, stride=2)
    
    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.conv_transpose2d(x, self.perm_weight, stride=2)


class CouplingBijection2D(nn.Module):
    """RealNVP coupling layer for 2D data with checkerboard masking.
    """
    def __init__(self, in_channels: int, hidden_channels: int = 64, n_hidden_layers: int = 1, mask_type: str = 'checkerboard', reverse_mask=False):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.n_hidden_layers = n_hidden_layers
        self.out_channels = self.in_channels
        self.mask_type = mask_type  # options: 'checkerboard' or 'channelwise'
        self.reverse_mask = reverse_mask
        
        # Neural network for computing scaling and translation parameters
        self.resnet = ResNet(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=in_channels * 2,  # scale and shift for each channel
            n_hidden_layers=n_hidden_layers
        )
        self.rescale = Rescale(in_channels)

    def _create_checkerboard_mask(self, shape: Tuple[int, int, int, int], device: torch.device) -> torch.Tensor:
        _, _, h, w = shape
        # Create base 2x2 checkerboard pattern
        base_mask = torch.zeros(1, 1, 2, 2, device=device)
        base_mask[0, 0, 0, 0] = 1
        base_mask[0, 0, 1, 1] = 1
        if self.reverse_mask:
            base_mask = 1 - base_mask
        
        # Calculate number of repetitions needed
        h_repeats = h // 2
        w_repeats = w // 2
        
        # Repeat the pattern to match input dimensions
        mask = base_mask.repeat(1, 1, h_repeats, w_repeats)
        return mask

    def _create_channelwise_mask(self, shape: Tuple[int, int, int, int], device: torch.device) -> torch.Tensor:
        _, c, h, w = shape
        # Create a channelwise mask where we either mask the first or last half of the channels
        mask = torch.cat([torch.ones(1, c//2, h, w, device=device), torch.zeros(1, c//2, h, w, device=device)], dim=1)
        if self.reverse_mask:
            mask = 1 - mask
        return mask
    
    def get_scale_and_shift(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Input shape: (batch_size, channels, height, width)
        if self.mask_type == 'checkerboard':
            mask = self._create_checkerboard_mask(x.shape, x.device)
        elif self.mask_type == 'channelwise':
            mask = self._create_channelwise_mask(x.shape, x.device)
        else:
            raise ValueError(f"Invalid mask type: {self.mask_type}")
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


class BlockBijection2D(nn.Module):
    """RealNVP block for 2D data with checkerboard masking.
    """
    def __init__(self, in_channels: int, hidden_channels: int = 64, n_hidden_layers: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.n_hidden_layers = n_hidden_layers
        self.out_channels = self.in_channels * 2  # half of channels continue, other half are done

        self.coupling_layers_checkerboard = nn.ModuleList([
            CouplingBijection2D(in_channels, hidden_channels, n_hidden_layers, 'checkerboard', reverse_mask=False),
            CouplingBijection2D(in_channels, hidden_channels, n_hidden_layers, 'checkerboard', reverse_mask=True),
            CouplingBijection2D(in_channels, hidden_channels, n_hidden_layers, 'checkerboard', reverse_mask=False)
        ])

        self.coupling_layers_channelwise = nn.ModuleList([
            CouplingBijection2D(4 * in_channels, 2 * hidden_channels, n_hidden_layers, 'channelwise', reverse_mask=False),
            CouplingBijection2D(4 * in_channels, 2 * hidden_channels, n_hidden_layers, 'channelwise', reverse_mask=True),
            CouplingBijection2D(4 * in_channels, 2 * hidden_channels, n_hidden_layers, 'channelwise', reverse_mask=False)
        ])

        self.squeeze = Squeeze()
        self.squeeze_permute = SqueezePermute(in_channels)
    
    def forward(self, x):
        ldj = 0

        for layer in self.coupling_layers_checkerboard:
            x, sldj = layer(x)
            ldj += sldj
        
        x = self.squeeze(x)

        for layer in self.coupling_layers_channelwise:
            x, sldj = layer(x)
            ldj += sldj

        x = self.squeeze.inverse(x)
        x = self.squeeze_permute(x)

        return x, ldj
    
    @torch.no_grad()
    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        for layer in reversed(self.coupling_layers_channelwise):
            z = layer.inverse(z)
        
        z = self.squeeze.inverse(z)

        for layer in reversed(self.coupling_layers_checkerboard):
            z = layer.inverse(z)

        return z


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

        # Create intermediate blocks
        self.blocks = self.build_intermediate_blocks(hidden_channels)

        # Create final block
        in_channels = self.blocks[-1].out_channels
        hidden_channels *= 2 ** len(self.blocks)
        self.final_layers = self.build_final_block(in_channels, hidden_channels, n_layers=4)

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
    
    def build_final_block(self, in_channels: int, hidden_channels: int, n_layers: int) -> nn.Module:
        layers = []
        reverse_mask = False
        for _ in range(n_layers):
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
        z = x
        total_log_det = torch.zeros(x.shape[0], device=x.device)

        for block in self.blocks:
            z, log_det = block(z)
            total_log_det += log_det

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
        for block in reversed(self.blocks):
            x = block.inverse(x)
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