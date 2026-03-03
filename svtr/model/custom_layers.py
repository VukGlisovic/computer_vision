from typing import Callable, Tuple
from itertools import product

import numpy as np
import torch
from torch import nn


class CBA(nn.Module):
    """Convolution -> BatchNorm -> Activation (CBA).
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 2,
                 padding: int = 1,
                 act: Callable = nn.GELU):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class LearnablePositionEmbedding(nn.Module):
    """
    Learnable position embedding of the SVTR architecture. The full weight matrix is built for
    in_h * max_in_w positions (maximum width). In forward, only the first nr_patches rows are
    used so that variable-width inputs are supported.
    """

    def __init__(self, in_h: int, max_in_w: int, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.in_h = in_h
        self.max_in_w = max_in_w
        self.num_embeddings = self.in_h * self.max_in_w

        self.pos_embedding = nn.Embedding(num_embeddings=self.num_embeddings, embedding_dim=embedding_dim)
        nn.init.trunc_normal_(self.pos_embedding.weight, std=0.02)  # Truncated normal as in ViT/BERT

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expected input shape: [bs, nr_patches, embedding_dim] (height and width flattened)
        nr_patches = x.shape[1]
        x = x + self.pos_embedding.weight[:nr_patches]
        return x


class SinusoidalPositionEmbedding(nn.Module):
    """Positional embedding based on sine and cosine waves.
    
    Uses fixed (non-learnable) sinusoidal encodings as described in
    "Attention Is All You Need" (Vaswani et al., 2017).

    The in_h of the feature map with an image height of 32, is 32/4=8 (SVTR downsamples 4
    times before the positional encoding is added). The width is variable, but also downsampled
    by a factor of 4. We use the width_multiplier to determine the number of horizontal
    positions per row.
    For example, with a feature map height of 8, the max supported feature map width would be
    width_multiplier * in_h = 48 * 8 = 384. Consequently, a feature map width of 384 results
    in supporting a max image width of 1536 pixels.
    With a feature map height of 12 (i.e. 48px image height), the max supported feature map
    width becomes 48 * 12 = 576. Thus, a max image width of 2304 pixels.

    With width_multiplier=48, we should have plenty of positions to cover pretty much all the
    incoming text recognition data.
    """

    def __init__(self, in_h: int, max_in_w: int, embedding_dim: int, base: int = 10000):
        super().__init__()
        self.in_h = in_h
        self.max_in_w = max_in_w
        self.embedding_dim = embedding_dim
        self.base = base

        # Precompute the scale for the positional encoding
        # This is to make sure the sinusiodal encoding (ranging from -1 to 1) is not too large and will not overpower the features.
        self.scale = embedding_dim ** 0.5

        # Precompute the sinusoidal positional encodings for max_positions
        self.max_total_positions = self.max_in_w * in_h
        pos_encoding = self._create_sinusoidal_encoding(self.max_total_positions)
        self.register_buffer('pos_encoding', pos_encoding)

    def _create_sinusoidal_encoding(self, num_total_positions: int) -> torch.Tensor:
        """Generate sinusoidal positional encodings."""
        position = torch.arange(num_total_positions, dtype=torch.float32).unsqueeze(1)  # shape (num_total_positions, 1)
        div_term = torch.exp(
            torch.arange(0, self.embedding_dim, 2, dtype=torch.float32) * (-np.log(self.base) / self.embedding_dim)
        )  # shape (dim/2,)
        angles = position * div_term  # shape (num_positions, embedding_dim/2)
        encodings = torch.zeros(num_total_positions, self.embedding_dim)  # shape (num_total_positions, dim)
        # Interleave sin and cos encodings
        encodings[:, 0::2] = torch.sin(angles)
        encodings[:, 1::2] = torch.cos(angles)
        return encodings

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """This layer expects to have the height and width channels flattened already.
        Expected input shape: [bs, nr_patches, embedding_dim]
        """
        nr_patches = x.shape[1]
        x = x * self.scale + self.pos_encoding[:nr_patches]
        return x


class WindowedMultiheadAttention(nn.Module):
    """Inspired by https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
    """

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 mixing_type: str = 'global',
                 in_h: int = None,
                 window_shape: Tuple[int, int] = (7, 11),
                 attn_dropout: float = 0.,
                 linear_dropout: float = 0.):
        super().__init__()
        # Some checks to make sure calculations are feasible
        assert embed_dim % num_heads == 0, "num_heads must be a divisor of embed_dim."
        assert mixing_type in ['local', 'global'], f"Unknown mixer '{mixing_type}'."
        assert in_h is not None, "You must provide an input height."
        assert (window_shape[0] % 2 == 1) and (window_shape[1] % 2 == 1), "Attention mask kernel must contain uneven numbers"

        # Save attributes
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mixing_type = mixing_type
        self.in_h = in_h
        self.window_shape = window_shape  # used only for mixing_type='local'
        self.attn_dropout = attn_dropout
        self.linear_dropout = linear_dropout

        # Create new attributes based on input configuration
        self.dim_one_head = embed_dim // num_heads
        self.scale = self.dim_one_head ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)  # create one big dense layer for query, key and value matrices for efficient computing
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.linear_dropout = nn.Dropout(linear_dropout)

    def create_local_attention_mask(self, in_h: int, in_w: int) -> torch.Tensor:
        # we only want to attend to a local regions. Therefore, we need to make sure far away patches get no attention.
        # this is achieved by basically setting the attention values of out of far away patches to -inf.
        kh, kw = self.window_shape  # kernel height, kernel width
        mask = np.full([in_h * in_w, in_h + kh - 1, in_w + kw - 1], -np.inf, dtype=np.float32)
        for h, w in product(range(0, in_h), range(0, in_w)):
            # for every location, create a mask pointing to the region around that location
            mask[h * in_w + w, h:h + kh, w:w + kw] = 0.
        # remove edges that are out of the image
        pad_h, pad_w = kh // 2, kw // 2
        mask = mask[:, pad_h: in_h + pad_h, pad_w: in_w + pad_w]
        # flatten attention mask for each location and prepend two dimensions to match attentions rank
        mask = mask.reshape((1, 1, in_h * in_w, -1))
        return torch.from_numpy(mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Note that the expected input shape should be flattened in the spatial resolution. I.e. x.shape = [bs, nr_patches, embed_dim]
        bs, nr_patches, _ = x.shape
        # Get the Q, K and V matrices
        QKV = self.qkv(x)
        # Reshape from [bs, nr_patches, 3*embed_dim] to [bs, nr_patches, QKV, nr_heads, dim_one_head]. Note embed_dim=nr_heads*dim_one_head and QKV=3
        QKV = QKV.reshape((bs, nr_patches, 3, self.num_heads, self.dim_one_head))
        # After permutation: [QKV, bs, nr_heads, nr_patches, dim_one_head]
        QKV = QKV.permute((2, 0, 3, 1, 4))
        q, k, v = QKV[0] * self.scale, QKV[1], QKV[2]
        # Calculate attentions
        attn = q.matmul(k.permute((0, 1, 3, 2)))  # [bs, num_heads, nr_patches, nr_patches]
        if self.mixing_type == 'local':
            # When looking at a local region around a location, we want to remove all attention outside of that region
            # We also need to handle flexible input size, so we need to recalculate the attention mask
            w = nr_patches // self.in_h
            mask = self.create_local_attention_mask(self.in_h, w)
            attn += mask.to(attn.device)
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        # Multiply the attentions with V (the value matrix)
        # Shape attn.matmul(v): [bs, num_heads, nr_patches, dim_one_head]
        # After permuting and reshaping, we basically concatenate back again the split up (across the heads) embedding vectors of each patch
        x = attn.matmul(v).permute((0, 2, 1, 3)).reshape((bs, nr_patches, self.embed_dim))
        x = self.proj(x)
        x = self.linear_dropout(x)
        return x
