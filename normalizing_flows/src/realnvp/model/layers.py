from typing import Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

from normalizing_flows.src.realnvp.model.resnet import ResNet
from normalizing_flows.src.realnvp.model.layer_utils import weight_norm


class PreprocessImages(nn.Module):
    """
    Preprocess the input images:
    1. Dequantization; make pixel values continuous (instead of descrete).
    2. Convert to logits.
    """

    def __init__(self, alpha=0.05):
        super(PreprocessImages, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x should have dynamic range [0, 1] with shape [bs, C, H, W].
        """
        constr = 1. - 2 * self.alpha
        y = (x * 255. + torch.rand_like(x)) / 256.  # Dequantization: add random uniform to make space were trying to model continuous
        y = (2 * y - 1) * constr  # Dynamic range [-constr, constr] = [-1 + 2*alpha, 1 - 2*alpha]
        y = (y + 1) / 2  # Dynamic range [(1-constr)/2, (1+constr)/2] = [alpha, 1 - alpha]
        y = y.log() - (1. - y).log()  # Take logit of y

        # Calculate log-determinant of Jacobian of this transform
        ldj = F.softplus(y) + F.softplus(-y) - F.softplus((1. - constr).log() - constr.log())
        sldj = ldj.view(ldj.size(0), -1).sum(-1)

        return y, sldj

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        # The inverse of the logit function, is the sigmoid function
        x = torch.sigmoid(y)
        return x


class Rescale(nn.Module):
    """
    Per-channel rescaling. Need a proper `nn.Module` so we can wrap it
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
    """
    Squeezes a [bs, C, H, W] shape tensor into a [bs, 4C, H/2, W/2] shape tensor.
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
    """
    Squeezes a [bs, C, H, W] shape tensor into a [bs, 4C, H/2, W/2] shape tensor
    but compared to Squeeze it also applies some shuffling.
    """

    def __init__(self, in_channels: int):
        super(SqueezePermute, self).__init__()
        self.in_channels = in_channels
        self.perm_weight = nn.Parameter(self._create_perm_weight(), requires_grad=False)
    
    def _create_perm_weight(self) -> torch.Tensor:
        c = self.in_channels
        # Define a permutation of input channels with shape is [4, 1, 2, 2]
        squeeze_matrix = torch.tensor(
            [[[[1., 0.], [0., 0.]]],
            [[[0., 0.], [0., 1.]]],
            [[[0., 1.], [0., 0.]]],
            [[[0., 0.], [1., 0.]]]]
        )
        perm_weight = torch.zeros((4 * c, c, 2, 2))  # shape [4c, c, 2, 2]
        # Insert the squeeze matrix into every channel index
        for c_idx in range(c):
            perm_weight[c_idx * 4: (c_idx + 1) * 4, c_idx: c_idx + 1, :, :] = squeeze_matrix
        # Shuffle the permutation matrix
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


class CheckerboardBijection2D(nn.Module):
    """
    Checkerboard coupling layer for 2D data.
    """

    def __init__(self, in_channels: int, hidden_channels: int = 64, n_residual_blocks: int = 1, reverse_mask=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = self.in_channels
        self.reverse_mask = reverse_mask
        
        # Neural network for computing scaling and translation parameters
        self.resnet = ResNet(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=in_channels * 2,  # scale and shift for each channel
            n_residual_blocks=n_residual_blocks,
            in_factor=2.  # checkerboard masks out half of the values for the conv layer, multiply by 2 to correct
        )
        self.rescale = weight_norm(Rescale(in_channels))

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
    
    def get_scale_and_shift(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Input shape: (batch_size, channels, height, width)
        mask = self._create_checkerboard_mask(x.shape, x.device)
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
        x = (z - t_masked) * torch.exp(-log_s_masked)
        
        return x


class ChannelwiseBijection2D(nn.Module):
    """
    Channelwise coupling layer for 2D data.
    """

    def __init__(self, in_channels: int, hidden_channels: int = 64, n_residual_blocks: int = 1, reverse_mask=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = self.in_channels
        self.reverse_mask = reverse_mask

        # Neural network for computing scaling and translation parameters
        self.resnet = ResNet(
            in_channels=in_channels // 2,  # chopping off half of the channels
            hidden_channels=hidden_channels,
            out_channels=in_channels,  # scale and shift for each value
            n_residual_blocks=n_residual_blocks,
            in_factor=1.  # no correction needed for channelwise masking
        )
        self.rescale = weight_norm(Rescale(in_channels // 2))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Implementation is with torch.chunk instead of with a mask as is described in
        the RealNVP paper to make this layer more efficient.
        """
        if self.reverse_mask:
            z_update, z_id = torch.chunk(x, 2, dim=1)
        else:
            z_id, z_update = torch.chunk(x, 2, dim=1)
        params = self.resnet(z_id)
        log_s, t = torch.chunk(params, 2, dim=1)  # chunk along channel dimension
        log_s = self.rescale(torch.tanh(log_s))

        # Apply transformation
        z_update = z_update * torch.exp(log_s) + t
        if self.reverse_mask:
            z = torch.cat([z_update, z_id], dim=1)
        else:
            z = torch.cat([z_id, z_update], dim=1)

        # Compute log determinant
        log_det = log_s.sum(dim=[1, 2, 3])

        return z, log_det

    @torch.no_grad()
    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        if self.reverse_mask:
            x_update, x_id = torch.chunk(z, 2, dim=1)
        else:
            x_id, x_update = torch.chunk(z, 2, dim=1)
        params = self.resnet(x_id)
        log_s, t = torch.chunk(params, 2, dim=1)  # chunk along channel dimension
        log_s = self.rescale(torch.tanh(log_s))

        # Apply inverse transformation
        x_update = (x_update - t) * torch.exp(-log_s)
        if self.reverse_mask:
            x = torch.cat([x_update, x_id], dim=1)
        else:
            x = torch.cat([x_id, x_update], dim=1)

        return x
