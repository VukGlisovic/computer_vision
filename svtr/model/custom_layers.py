from itertools import product
import numpy as np
import torch
from torch import nn


class CBA(nn.Module):
    """Convolution -> BatchNorm -> Activation (CBA).
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


class WindowedMultiheadAttention(nn.Module):
    """Inspired by https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
    """

    def __init__(self,
                 embed_dim=128,
                 num_heads=8,
                 mixing_type='global',
                 in_hw=None,
                 window_shape=[7, 11],
                 attn_dropout=0.,
                 linear_dropout=0.):
        super().__init__()
        # some checks to make sure calculations are feasible
        assert embed_dim % num_heads == 0, "num_heads must be a divisor of embed_dim."
        assert mixing_type in ['local', 'global'], f"Unknown mixer '{mixing_type}'."
        assert in_hw is not None, "You must provide an input shape."
        assert (window_shape[0] % 2 == 1) and (window_shape[1] % 2 == 1), "Attention mask kernel must contain uneven numbers"
        # save attributes
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mixing_type = mixing_type
        self.in_hw = in_hw
        self.window_shape = window_shape  # used only for mixing_type='local'
        self.attn_dropout = attn_dropout
        self.linear_dropout = linear_dropout
        # create new attributes based on input configuration
        self.scale = embed_dim ** -0.5
        self.dim_one_head = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)  # create one big dense layer for query, key and value matrices for efficient computing
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.linear_dropout = nn.Dropout(linear_dropout)
        H, W = in_hw
        if mixing_type.lower() == 'local':
            # we only want to attend to a local regions. Therefore, we need to make sure far away patches get no attention.
            # this is achieved by basically setting the attention values of out of far away patches to -inf.
            kh, kw = window_shape  # kernel height, kernel width
            mask = np.full([H * W, H + kh - 1, W + kw - 1], -np.inf, dtype=np.float32)
            for h, w in product(range(0, H), range(0, W)):
                # for every location, create a mask pointing to the region around that location
                mask[h * W + w, h:h + kh, w:w + kw] = 0.
            # remove edges that are out of the image
            pad_h, pad_w = kh // 2, kw // 2
            mask = mask[:, pad_h: H+pad_h, pad_w: W+pad_w]
            # flatten attention mask for each location and prepend two dimensions to match attentions rank
            mask = mask.reshape((1, 1, H*W, -1))
            self.mask = torch.from_numpy(mask)
            self.mask = nn.Parameter(self.mask, requires_grad=False)

    def forward(self, x):
        bs = x.shape[0]  # batch size
        # get the Q, K and V matrices
        QKV = self.qkv(x)
        # reshape from [bs, nr patches, 3*embed_dim] to [bs, nr patches, 3, nr heads, dim one head]. Note embed_dim=nr_heads*dim_one_head
        QKV = QKV.reshape((bs, -1, 3, self.num_heads, self.dim_one_head))
        # after permutation: [QKV, bs, nr heads, nr patches, dim one head]
        QKV = QKV.permute((2, 0, 3, 1, 4))
        q, k, v = QKV[0] * self.scale, QKV[1], QKV[2]
        # calculate attentions
        attn = (q.matmul(k.permute((0, 1, 3, 2))))
        if self.mixing_type == 'local':
            # when looking at a local region around a location, we want to remove all attention outside of that region
            attn += self.mask
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        # multiply the attentions with V (the value matrix)
        x = (attn.matmul(v)).permute((0, 2, 1, 3)).reshape((bs, -1, self.embed_dim))
        x = self.proj(x)
        x = self.linear_dropout(x)
        return x
