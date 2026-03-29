from typing import Callable, Tuple, List

import numpy as np
import torch
from torch import nn

from svtr.model.custom_layers import CBA, WindowedMultiheadAttention


class PatchEmbedding(nn.Module):
    """
    The first block of the SVTR architecture. It embeds the image into a sequence of patches.
    """

    def __init__(self,
                 in_c: int,
                 in_h: int,
                 max_in_w: int,
                 hdim1: int = 256,
                 hdim2: int = 512):
        super().__init__()
        self.in_c = in_c
        self.in_h = in_h  # Height is deterministic, but the width is variable, so we don't store the width
        self.max_in_w = max_in_w
        self.hdim1 = hdim1
        self.hdim2 = hdim2

        self.out_c: int = self.hdim2
        self.out_h: int = self.in_h // 4
        self.max_out_w: int = int(np.ceil(self.max_in_w / 4))

        self.cba1 = CBA(
            in_channels=self.in_c,
            out_channels=self.hdim1,
            kernel_size=3,
            stride=2,
            padding=1,
            act=nn.GELU
        )
        self.cba2 = CBA(
            in_channels=self.hdim1,
            out_channels=self.hdim2,
            kernel_size=3,
            stride=2,
            padding=1,
            act=nn.GELU
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cba1(x)
        x = self.cba2(x)
        x = x.flatten(start_dim=2, end_dim=3)  # flatten sothat later we can easily add a position embedding
        x = x.permute((0, 2, 1))  # out shape: [bs, nr_patches, hdim2]
        return x


class MLP(nn.Module):
    """
    Simple two-layer MLP block.
    """

    def __init__(self, in_dim: int, hidden_dim_factor: int = 2, act: Callable = nn.GELU, dropout: float = 0.):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim_factor = hidden_dim_factor
        self.dropout = dropout

        out_dim = in_dim
        hidden_dim = int(in_dim * hidden_dim_factor)
        self.dense1 = nn.Linear(in_dim, hidden_dim)
        self.act = act()
        self.dense2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.dense2(x)
        x = self.drop(x)
        return x


class MixingBlock(nn.Module):
    """
    Mixing block of the SVTR architecture. It consists of a multi-head attention layer and a MLP layer.
    """

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 mixing_type: str = 'Global',
                 window_shape: Tuple[int, int] = [7, 11],
                 in_h: int = None,
                 mlp_hidden_dim_factor: int = 2,
                 attn_dropout: float = 0.,
                 linear_dropout: float = 0.,
                 act: Callable = nn.GELU):
        super().__init__()
        # part 1 of mixing block: multi-head attention
        self.norm1 = nn.LayerNorm(embed_dim)
        self.windowed_mha = WindowedMultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            mixing_type=mixing_type,
            in_h=in_h,
            window_shape=window_shape,
            attn_dropout=attn_dropout,
            linear_dropout=linear_dropout
        )
        # part 2 of mixing block: multi-layer perceptron
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(in_dim=embed_dim, hidden_dim_factor=mlp_hidden_dim_factor, act=act, dropout=linear_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.windowed_mha(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SequentialMixingBlocks(nn.Module):
    """
    Sequential mixing blocks of the SVTR architecture. It consists of a list of mixing blocks.
    """

    def __init__(self,
                 embed_dim: int,
                 out_dim: int,
                 num_heads,
                 mixing_type_list: List[str] = None,
                 window_shape: Tuple[int, int] = [7, 11],
                 in_h: int = None,
                 mlp_hidden_dim_factor: int = 2,
                 attn_dropout: float = 0.,
                 linear_dropout: float = 0.,
                 act: Callable = nn.GELU):
        super().__init__()
        assert isinstance(mixing_type_list, list), "You must provide a list of mixing block types."
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.mixers = mixing_type_list
        self.in_h = in_h

        self.mixing_blocks = nn.ModuleList()
        for mixer in mixing_type_list:
            block = MixingBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mixing_type=mixer,
                window_shape=window_shape,
                in_h=in_h,
                mlp_hidden_dim_factor=mlp_hidden_dim_factor,
                attn_dropout=attn_dropout,
                linear_dropout=linear_dropout,
                act=act
            )
            self.mixing_blocks.append(block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.mixing_blocks:
            x = b(x)
        return x


class MixingBlocksMerging(SequentialMixingBlocks):
    """
    Mixing blocks merging of the SVTR architecture. It consists of a list of mixing blocks
    followed by a merging convolution to subsample in the height dimension.
    """

    def __init__(self,
                 embed_dim: int,
                 out_dim: int,
                 num_heads: int,
                 mixing_type_list: List[str] = None,
                 window_shape: Tuple[int, int] = (7, 11),
                 in_h: int = None,
                 mlp_hidden_dim_factor: int = 2,
                 attn_dropout: float = 0.,
                 linear_dropout: float = 0.,
                 act: Callable = nn.GELU):
        super().__init__(embed_dim, out_dim, num_heads, mixing_type_list, window_shape, in_h, mlp_hidden_dim_factor, attn_dropout, linear_dropout, act)
        self.out_h: int = in_h // 2

        self.merging_conv = nn.Conv2d(
            in_channels=embed_dim,
            out_channels=out_dim,
            kernel_size=3,
            stride=(2, 1),
            padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # mixing blocks: local/global multi-head attention layers
        x = super().forward(x)
        # shape transformations before merging_conv: [bs, nr_patches, embed_dim] -> [bs, embed_dim, nr_patches] -> [bs, embed_dim, height, width]
        x = x.permute([0, 2, 1])
        x = x.reshape([x.shape[0], self.embed_dim, self.in_h, -1])
        # merging: subsample in the height dimension
        x = self.merging_conv(x)
        # shape transformations: [bs, embed_dim, height/2, width] -> [bs, embed_dim, nr_patches] -> [bs, nr_patches, embed_dim]
        x = x.reshape((x.shape[0], self.out_dim, -1))
        x = x.permute((0, 2, 1))
        return x


class MixingBlocksCombining(SequentialMixingBlocks):
    """
    Mixing blocks combining of the SVTR architecture. It consists of a list of mixing blocks
    followed by a dense layer and a dropout layer.
    """

    def __init__(self,
                 embed_dim: int,
                 out_dim: int,
                 num_heads: int,
                 mixing_type_list: List[str] = None,
                 window_shape: Tuple[int, int] = (7, 11),
                 in_h: int = None,
                 mlp_hidden_dim_factor: int = 2,
                 attn_dropout: float = 0.,
                 linear_dropout: float = 0.,
                 last_dropout: float = 0.,
                 act: Callable = nn.GELU):
        super().__init__(embed_dim, out_dim, num_heads, mixing_type_list, window_shape, in_h, mlp_hidden_dim_factor, attn_dropout, linear_dropout, act)
        self.last_dropout = last_dropout
        self.out_h: int = 1  # because of pooling over height axis

        self.dense = nn.Linear(
            in_features=embed_dim,
            out_features=out_dim
        )
        self.act = nn.Hardswish()
        self.dropout = nn.Dropout(last_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # mixing blocks: local/global multi-head attention layers
        x = super().forward(x)
        # shape transformations before combining: [bs, nr_patches, embed_dim] -> [bs, embed_dim, nr_patches] -> [bs, embed_dim, height, width]
        x = x.reshape([x.shape[0], self.in_h, -1, self.embed_dim])
        # combining: pool over height dimension and apply transformations
        x = x.mean(axis=1, keepdim=False)  # pooling height dim; out shape [bs, width, embed_dim]
        x = self.dense(x)  # out shape [bs, width, out_dim]
        x = self.act(x)
        x = self.dropout(x)
        return x
