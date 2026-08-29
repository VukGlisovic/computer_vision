import math
from typing import List, Tuple

import torch
from torch import nn

from srec.model.srec_loss import non_shared_get_Kp


def conv(in_ch: int,
         out_ch: int,
         kernel_size: int,
         bias: bool = True,
         dilation: int = 1,
         stride: int = 1) -> nn.Conv2d:
    """2D convolution that keeps spatial size when stride=1.

    The padding is selected as kernel_size // 2 for the dense case and as
    dilation for dilated convolutions, matching the convention used by
    the upstream EDSR implementation.
    """
    padding = kernel_size // 2 if dilation == 1 else dilation
    return nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, dilation=dilation, padding=padding, bias=bias)


def get_act(act: str, n_feats: int = 0) -> nn.Module:
    """Return the activation module matching `act`.
    """
    if act == "relu":
        return nn.ReLU(inplace=True)
    if act == "prelu":
        return nn.PReLU(n_feats)
    if act == "leaky_relu":
        return nn.LeakyReLU(inplace=True)
    if act == "none":
        return nn.Identity()
    raise NotImplementedError(f"{act} is not implemented")


class ResidualBlock(nn.Module):
    """Residual block with the following structure if batch norm in use:
    conv -> bn -> act -> (dilated) conv -> bn -> + -> out
    """

    def __init__(self,
                 n_feats: int,
                 kernel_size: int,
                 act: str = "leaky_relu",
                 dilation_rate: int = 1,
                 bn: bool = False) -> None:
        super().__init__()

        layers: List[nn.Module] = []
        for i in range(2):
            dr = 1 if i == 0 else dilation_rate
            layers.append(conv(n_feats, n_feats, kernel_size, dilation=dr, bias=True))
            if bn:
                layers.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                layers.append(get_act(act))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.block(x)
        res = res + x
        return res


class PixelShuffleUpsampler(nn.Sequential):
    """Pixel-shuffle based upsampler for power-of-two scale factors.

    For scale = 2^n, the upsampler is built as n repeated (conv -> pixel_shuffle(2)) stages.
    """

    def __init__(self,
                 scale: int,
                 n_feats: int,
                 bn: bool = False,
                 act: str = "none",
                 bias: bool = True) -> None:
        layers: List[nn.Module] = []
        assert (scale & (scale - 1)) == 0, f"{scale} is not a power of 2. Only power-of-two scale factors are currently supported."
        self.n_stages = int(math.log(scale, 2))  # When scale=2, then there's only one stage.
        for _ in range(self.n_stages):
            layers.append(conv(in_ch=n_feats, out_ch=4*n_feats, kernel_size=3, bias=bias))
            layers.append(nn.PixelShuffle(2))
            if bn:
                layers.append(nn.BatchNorm2d(n_feats))
            layers.append(get_act(act))
        super().__init__(*layers)


class EDSRDec(nn.Module):
    """EDSR-style (Enhanced Deep Super-Resolution) decoder with a head, residual body
    and configurable tail.
    It is based on the 2017 paper "Enhanced Deep Residual Networks for Single Image
    Super-Resolution" by Lim et al. (CVPR NTIRE 2017 winner), https://arxiv.org/abs/1707.02921.

    The head is a 1x1 projection from in_channels to out_channels.
    The body is a stack of `resblocks` residual blocks followed by a final convolution.
    The tail selects how the decoder finishes:
    * `conv` - a 1x1 convolution (keeps spatial size).
    * `none` - identity.
    * `upsample` - a 2x Upsampler.

    forward additionally accepts a `features_to_fuse` tensor that is added to the head
    output, which is how the auto-regressive context from a previous scale is injected
    into the next scale.
    """

    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 resblocks: int = 8,
                 kernel_size: int = 3,
                 tail: str = "none") -> None:
        super().__init__()
        self.head = conv(in_ch, out_ch, 1)

        m_body: List[nn.Module] = [ResidualBlock(out_ch, kernel_size) for _ in range(resblocks)]
        m_body.append(conv(out_ch, out_ch, kernel_size))
        self.body = nn.Sequential(*m_body)

        self.tail: nn.Module
        if tail == "conv":
            self.tail = conv(out_ch, out_ch, 1)
        elif tail == "none":
            self.tail = nn.Identity()
        elif tail == "upsample":
            self.tail = PixelShuffleUpsampler(scale=2, n_feats=out_ch)
        else:
            raise NotImplementedError(f"{tail} is not implemented.")

    def forward(self, x: torch.Tensor, features_to_fuse: torch.Tensor = 0.) -> torch.Tensor:
        x = self.head(x)
        x = x + features_to_fuse
        x = self.body(x) + x
        x = self.tail(x)
        return x


class StackedDilatedConvs(nn.Module):
    """Parallel dilated convolutions whose outputs are concatenated and fused.

    Each dilation rate in `dilation_rates` (a tuple of integers e.g. (1, 2, 4))
    produces a feature map at the same spatial resolution. The maps are concatenated
    along the channel dimension and a final 1x1 convolution projects them to `out_ch`
    channels.
    """

    def __init__(self,
                 dilation_rates: Tuple[int, ...],
                 in_ch: int,
                 out_ch: int,
                 bias: bool = True,
                 kernel_size: int = 3) -> None:
        super().__init__()
        self.dilated_convs = nn.ModuleList([conv(in_ch, in_ch, kernel_size, dilation=dilation) for dilation in dilation_rates])
        self.conv_1x1 = conv(len(dilation_rates) * in_ch, out_ch, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([dilated_conv(x) for dilated_conv in self.dilated_convs], dim=1)
        x = self.conv_1x1(x)
        return x


class DilatedConvsProbabilityClassifier(nn.Module):
    """Predict `Kp` logistic-mixture parameters per spatial position.

    Given a feature map of shape N x in_ch x H x W this module produces a tensor
    of shape N x Kp x H x W where Kp = num_params * C * K. The output is consumed
    by DiscretizedMixLogisticLoss to obtain per-pixel log-likelihoods.

    Args:
        in_ch: number of input channels.
        C: number of image channels (always 3 for RGB).
        num_params: number of distribution parameters per sub-pixel (always 4 for RGB).
            The generation of the 4 distribution parameters is done in the PixelCNN++ paper.
        K: number of mixture components.
        kernel_size: kernel size of the convolutions.
        dilation_rates: convolution dilation rates.
    """

    def __init__(self,
                 in_ch: int,
                 C: int,
                 num_params: int,
                 K: int = 10,
                 kernel_size: int = 3,
                 dilation_rates: Tuple[int, ...] = (1, 2, 4)) -> None:
        super().__init__()
        Kp = non_shared_get_Kp(K, C, num_params)
        self.stacked_dilated_conv_blocks = StackedDilatedConvs(dilation_rates, in_ch, Kp, kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stacked_dilated_conv_blocks(x)
