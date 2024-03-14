from torch import nn


class CBABlock(nn.Module):
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


class PatchEmbedding(nn.Module):

    def __init__(self, image_shape, hdim1=256, hdim2=512):
        super().__init__()
        self.image_shape = image_shape
        self.in_c, self.in_h, self.in_w = image_shape
        self.hdim1 = hdim1
        self.hdim2 = hdim2

        # outH = (inH - k + p) / s + 1
        # E.g. 16 = (32 - 3 + 1) / 2 + 1
        self.out_c = self.hdim2
        self.out_h = self.in_h // 4
        self.out_w = self.in_w // 4
        self.nr_patches = self.out_h * self.out_w

        self.cba1 = CBABlock(
            in_channels=self.in_c,
            out_channels=self.hdim1,
            kernel_size=3,
            stride=2,
            padding=1,
            act=nn.GELU
        )
        self.cba2 = CBABlock(
            in_channels=self.hdim1,
            out_channels=self.hdim2,
            kernel_size=3,
            stride=2,
            padding=1,
            act=nn.GELU
        )

    def forward(self, x):
        x = self.cba1(x)
        x = self.cba2(x)
        x = x.flatten(start_dim=2, end_dim=3)  # flatten sothat later we can easily add a position embedding
        x = x.permute((0, 2, 1))  # out shape: [bs, nr_patches, hdim2]
        return x
