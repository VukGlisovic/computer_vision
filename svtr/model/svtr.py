from typing import Tuple, Callable, Dict, Any

import torch
from torch import nn

from svtr.model import custom_blocks
from svtr.model import custom_layers


class SVTR(nn.Module):
    """
    SVTR implementation as described in the paper
    "SVTR: Scene Text Recognition with a Single Visual Model"
    Paper: https://arxiv.org/pdf/2205.00159

    Args:
        model_size: Select the model size from the following options: tiny, small, base, large
        image_shape:
        characters_path: Path to the characters.csv file. This file contains the characters that
            the model will be able to predict.
        positional_embedding: Select the positional embedding from the following options: learnable, sinusoidal or None.
        mlp_hidden_dim_factor: Increases the hidden dimension of the MLP by this factor.
        linear_dropout: Dropout rate for the dense layers in the MLPs and the attention layers.
        attn_dropout: Dropout rate for the attention weights.
        last_dropout: Last dropout rate for the dense layer before the output layer.
        act: Activation function for the MLPs.
    """

    TINY = {
        'embed_dim': [64, 128, 256],
        'out_dim': 192,
        'stages': [['local'] * 3, ['local'] * 3 + ['global'] * 3, ['global'] * 3],
        'num_heads': [2, 4, 8]
    }

    SMALL = {
        'embed_dim': [96, 192, 256],
        'out_dim': 192,
        'stages': [['local'] * 3, ['local'] * 5 + ['global'], ['global'] * 6],
        'num_heads': [3, 6, 8]
    }

    BASE = {
        'embed_dim': [128, 256, 384],
        'out_dim': 256,
        'stages': [['local'] * 3, ['local'] * 5 + ['global'], ['global'] * 9],
        'num_heads': [4, 8, 12]
    }

    LARGE = {
        'embed_dim': [192, 256, 512],
        'out_dim': 384,
        'stages': [['local'] * 3, ['local'] * 7 + ['global'] * 2, ['global'] * 9],
        'num_heads': [6, 8, 16]
    }

    CUSTOM = {
        'embed_dim': [64, 128, 256],
        'out_dim': 192,
        'stages': [['global']*2, ['global']*2, ['global']*2],
        'num_heads': [2, 4, 8]
    }

    def __init__(self,
                 model_size: str = 'tiny',
                 image_shape: Tuple[int, int, int] = [3, 32, 1000],
                 positional_embedding: str = 'sinusoidal',
                 mlp_hidden_dim_factor: int = 2,
                 linear_dropout: float = 0.,
                 attn_dropout: float = 0.,
                 last_dropout: float = 0.1,
                 act: Callable = nn.GELU,
                 vocab_size=11):
        super().__init__()
        self.model_size = model_size
        self.in_c, self.in_h, self.max_in_w = image_shape
        self.positional_embedding = positional_embedding
        self.vocab_size = vocab_size

        self.architecture_config: Dict[str, Any] = getattr(self, model_size.upper())
        ac = self.architecture_config  # Create acronym since we'll be using this config throughout the code below

        self.patch_embedding = custom_blocks.PatchEmbedding(in_c=self.in_c, in_h=self.in_h, max_in_w=self.max_in_w, hdim1=ac['embed_dim'][0] // 2, hdim2=ac['embed_dim'][0])
        self.window_shape = self._get_window_shape(self.patch_embedding.out_h)

        if positional_embedding == 'learnable':
            self.pos_emb = custom_layers.LearnablePositionEmbedding(in_h=self.patch_embedding.out_h, max_in_w=self.patch_embedding.max_out_w, embedding_dim=self.patch_embedding.hdim2)
        elif positional_embedding == 'sinusoidal':
            self.pos_emb = custom_layers.SinusoidalPositionEmbedding(in_h=self.patch_embedding.out_h, max_in_w=self.patch_embedding.max_out_w, embedding_dim=self.patch_embedding.hdim2)
        else:
            self.pos_emb = None

        self.stage1 = custom_blocks.MixingBlocksMerging(
            embed_dim=ac['embed_dim'][0],
            out_dim=ac['embed_dim'][1],
            num_heads=ac['num_heads'][0],
            mixing_type_list=ac['stages'][0],
            window_shape=self.window_shape,
            in_h=self.patch_embedding.out_h,
            mlp_hidden_dim_factor=mlp_hidden_dim_factor,
            attn_dropout=attn_dropout,
            linear_dropout=linear_dropout,
            act=act
        )

        self.stage2 = custom_blocks.MixingBlocksMerging(
            embed_dim=ac['embed_dim'][1],
            out_dim=ac['embed_dim'][2],
            num_heads=ac['num_heads'][1],
            mixing_type_list=ac['stages'][1],
            window_shape=self.window_shape,
            in_h=self.stage1.out_h,
            mlp_hidden_dim_factor=mlp_hidden_dim_factor,
            attn_dropout=attn_dropout,
            linear_dropout=linear_dropout,
            act=act
        )

        self.stage3 = custom_blocks.MixingBlocksCombining(
            embed_dim=ac['embed_dim'][2],
            out_dim=ac['out_dim'],
            num_heads=ac['num_heads'][2],
            mixing_type_list=ac['stages'][2],
            window_shape=self.window_shape,
            in_h=self.stage2.out_h,
            mlp_hidden_dim_factor=mlp_hidden_dim_factor,
            attn_dropout=attn_dropout,
            linear_dropout=linear_dropout,
            last_dropout=last_dropout,
            act=act
        )

        self.dense_out = nn.Linear(
            in_features=ac['out_dim'],
            out_features=vocab_size
        )

    @staticmethod
    def _get_window_shape(in_h: int) -> Tuple[int, int]:
        """
        Get the window shape for the local attention mask.

        In the original SVTR paper, the window shape is [7, 11] (which will follow from in_h=8).
        However, we want to make sure that bigger input heights will result in bigger window shapes.
        The formula in this method results in the following window shapes:
        img_height=32 -> in_h = 8 -> window_shape = [7, 11]
        img_height=48 -> in_h = 12 -> window_shape = [11, 17]
        img_height=64 -> in_h = 16 -> window_shape = [15, 23]
        """
        # Determine the window shape for the local attention mask
        vertical_size = in_h
        horizontal_size = round(1.5 * in_h)
        # Make sure the window shape is odd as that is required for the windowed attention layer
        if vertical_size % 2 == 0:
            vertical_size -= 1
        if horizontal_size % 2 == 0:
            horizontal_size -= 1
        window_shape = [vertical_size, horizontal_size]
        return window_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # embed image into patches with conv and batchnorm layers
        cc0 = self.patch_embedding(x)  # out shape: [bs, nr_patches, C0] where nr_patches = H/4 * W/4
        # add positional embedding
        if self.pos_emb is not None:
            cc0 = self.pos_emb(cc0)  # out shape: [bs, nr_patches, C0] (we keep the nr_patches flattened)
        # mixing and merging stage 1
        cc1 = self.stage1(cc0)  # out shape: [bs, nr_patches/2, C1] where nr_patches/2 = H/8 * W/4
        # mixing and merging stage 2
        cc2 = self.stage2(cc1)  # out shape: [bs, nr_patches/4, C2] where nr_patches/4 = H/16 * W/4
        # mixing and combining stage 3
        c = self.stage3(cc2)  # out shape: [bs, nr_patches_height_flattened, out_dim] where nr_patches_height_flattened = 1 * W/4
        # final dense transforming into character predictions for every vertical patch
        out = self.dense_out(c)  # out shape: [bs, nr_patches_height_flattened, nr_characters]
        out = out.log_softmax(axis=2)
        return out
