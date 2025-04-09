import torch
import torch.nn as nn

from normalizing_flows.src.realnvp.model.layers import CouplingBijection2D, Squeeze, SqueezePermute


class BlockBijection2D(nn.Module):
    """RealNVP block for 2D data with checkerboard masking.
    """
    def __init__(self, in_channels: int, hidden_channels: int = 32, n_cb_bijections: int = 3, n_cw_bijections: int = 3, n_residual_blocks: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = self.in_channels * 2  # half of channels continue, other half are done

        # Create checkerboard coupling layers
        coupling_layers_checkerboard = []
        reverse_mask = False
        for i in range(n_cb_bijections):
            coupling_layers_checkerboard.append(
                CouplingBijection2D(in_channels, hidden_channels, n_residual_blocks, 'checkerboard', reverse_mask=reverse_mask)
            )
            reverse_mask = not reverse_mask
        self.coupling_layers_checkerboard = nn.ModuleList(coupling_layers_checkerboard)

        # Create channelwise coupling layers
        coupling_layers_channelwise = []
        reverse_mask = False
        for i in range(n_cw_bijections):
            coupling_layers_channelwise.append(
                CouplingBijection2D(4 * in_channels, 2 * hidden_channels, n_residual_blocks, 'channelwise', reverse_mask=reverse_mask)
            )
            reverse_mask = not reverse_mask
        self.coupling_layers_channelwise = nn.ModuleList(coupling_layers_channelwise)

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
        z = self.squeeze_permute.inverse(z)
        z = self.squeeze(z)

        for layer in reversed(self.coupling_layers_channelwise):
            z = layer.inverse(z)
        
        z = self.squeeze.inverse(z)

        for layer in reversed(self.coupling_layers_checkerboard):
            z = layer.inverse(z)

        return z
