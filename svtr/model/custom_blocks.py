from torch import nn


class CBALayer(nn.Module):
    """Convolution, BatchNorm, Activation (CBA) block.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=2,
                 padding=1,
                 act=nn.GELU):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = act()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x
