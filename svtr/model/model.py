import torch
from torch import nn

from svtr.model import custom_blocks


config_tiny = {
    'embed_dim': [64, 128, 256],
    'out_dim': 192,
    'stages': [['local']*3, ['local']*3 + ['global']*3, ['global']*3],
    'num_heads': [2, 4, 8],
    'local_mixer': [[7, 11], [7, 11], [7, 11]]
}


config_small = {
    'embed_dim': [96, 192, 256],
    'out_dim': 192,
    'stages': [['local']*3, ['local']*5 + ['global'], ['global']*6],
    'num_heads': [3, 6, 8],
    'local_mixer': [[7, 11], [7, 11], [7, 11]]
}


config_base = {
    'embed_dim': [128, 256, 384],
    'out_dim': 256,
    'stages': [['local']*3, ['local']*5 + ['global'], ['global']*9],
    'num_heads': [4, 8, 12],
    'local_mixer': [[7, 11], [7, 11], [7, 11]]
}


config_large = {
    'embed_dim': [192, 256, 512],
    'out_dim': 384,
    'stages': [['local']*3, ['local']*7 + ['global']*2, ['global']*9],
    'num_heads': [6, 8, 16],
    'local_mixer': [[7, 11], [7, 11], [7, 11]]
}


class SVTR(nn.Module):

    def __init__(
            self,
            architecture='tiny',
            img_shape=[3, 32, 100],
            mlp_ratio=4,
            drop_rate=0.,
            last_drop=0.1,
            attn_drop_rate=0.,
            out_channels=192,
            vocab_size=11,
            act=nn.GELU):
        super().__init__()
        self.architecture = architecture
        self.config = eval(f'config_{architecture}')
        self.img_shape = img_shape
        self.out_channels = out_channels
        self.vocab_size = vocab_size

        self.patch_embedding = custom_blocks.PatchEmbedding(image_shape=self.img_shape, hdim1=self.config['embed_dim'][0] // 2, hdim2=self.config['embed_dim'][0])
        self.emb_indices = torch.arange(0, self.patch_embedding.nr_patches, dtype=torch.int32)
        self.emb_indices = nn.Parameter(self.emb_indices, requires_grad=False)
        self.pos_embedding = nn.Embedding(num_embeddings=self.patch_embedding.nr_patches, embedding_dim=self.patch_embedding.hdim2)

        self.stage1 = custom_blocks.MixingBlocksMerging(
            embed_dim=self.config['embed_dim'][0],
            out_dim=self.config['embed_dim'][1],
            num_heads=self.config['num_heads'][0],
            mixing_type_list=self.config['stages'][0],
            window_shape=self.config['local_mixer'][0],
            in_hw=[self.patch_embedding.out_h, self.patch_embedding.out_w],
            mlp_hidden_dim_factor=mlp_ratio,
            attn_dropout=attn_drop_rate,
            linear_dropout=drop_rate,
            act=act
        )

        self.stage2 = custom_blocks.MixingBlocksMerging(
            embed_dim=self.config['embed_dim'][1],
            out_dim=self.config['embed_dim'][2],
            num_heads=self.config['num_heads'][1],
            mixing_type_list=self.config['stages'][1],
            window_shape=self.config['local_mixer'][1],
            in_hw=[self.stage1.out_h, self.stage1.out_w],
            mlp_hidden_dim_factor=mlp_ratio,
            attn_dropout=attn_drop_rate,
            linear_dropout=drop_rate,
            act=act
        )

        self.stage3 = custom_blocks.MixingBlocksCombining(
            embed_dim=self.config['embed_dim'][2],
            out_dim=self.config['out_dim'],
            num_heads=self.config['num_heads'][2],
            mixing_type_list=self.config['stages'][2],
            window_shape=self.config['local_mixer'][2],
            in_hw=[self.stage2.out_h, self.stage2.out_w],
            mlp_hidden_dim_factor=mlp_ratio,
            attn_dropout=attn_drop_rate,
            linear_dropout=drop_rate,
            out_drop=last_drop,
            act=act
        )

        self.dense_out = nn.Linear(
            in_features=self.config['out_dim'],
            out_features=vocab_size
        )

    def forward(self, x):
        # embed image into patches with conv and batchnorm layers
        cc0 = self.patch_embedding(x)
        # add positional embedding
        x = cc0 + self.pos_embedding(self.emb_indices)
        # mixing and merging stage 1
        cc1 = self.stage1(x)
        # mixing and merging stage 2
        cc2 = self.stage2(cc1)
        # mixing and combining stage 3
        c = self.stage3(cc2)
        # final dense transforming into character predictions for every vertical patch
        return self.dense_out(c)


def print_model_parameters(model):
    nr_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    nr_non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Number of trainable/non-trainable parameters: {nr_trainable_params:,} / {nr_non_trainable_params:,}")


def save_model(model, path):
    torch.save(model, path)


def load_model(path):
    return torch.load(path)
