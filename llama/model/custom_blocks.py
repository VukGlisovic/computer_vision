from torch import nn
from llama.model.custom_layers import RMSNorm, RoPEMaskedMultiheadAttention, SwiGLU


class LlamaBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.rms = RMSNorm((config['context_window'], config['d_model']))
        self.rope_attention = RoPEMaskedMultiheadAttention(config)
        self.feedforward = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']),
            SwiGLU(config['d_model']),
        )

    def forward(self, x):
        """One Llama Block.
        """
        x = self.rms(x)  # RMS pre-normalization
        x = x + self.rope_attention(x)  # residual connection
        x = self.rms(x)  # RMS pre-normalization
        x = x + self.feedforward(x)  # residual connection
        return x
