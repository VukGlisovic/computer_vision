from collections import OrderedDict
import torch
from torch import nn
from torch.nn import functional as F

from llama.model.custom_layers import SwiGLU
from llama.model.custom_blocks import LlamaBlock


class Llama(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = self.config['vocab_size']
        self.context_window = self.config['context_window']
        # embedding layer for token index to vector representation
        self.embeddings = nn.Embedding(config['vocab_size'], config['d_model'])
        # multiple LlamaBlocks sequentially
        self.llama_blocks = nn.Sequential(
            OrderedDict([(f"llama_{i}", LlamaBlock(config)) for i in range(config['n_layers'])])
        )
        # feedforward network (FFN) for final output
        self.ffn = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']),
            SwiGLU(config['d_model']),
            nn.Linear(config['d_model'], config['vocab_size']),
        )

    def forward(self, indices, targets=None):
        # input token indices to embedding vectors
        x = self.embeddings(indices)
        # process the embeddings through the LlamaBlocks
        x = self.llama_blocks(x)
        # pass the processed input through the final FFN for output logits
        logits = self.ffn(x)

        if targets is None:
            return logits
        else:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
            return logits, loss

    # Generate function for text generation using the trained model
    def generate(self, device, tokenizer=None, tk_kwargs=None, max_new_tokens=30):
        indices = torch.zeros(1, 1).long().to(device)  # shape (batch dim, characters dim)
        for _ in range(max_new_tokens):
            logits = self(indices[:, -self.context_window:])  # model inference for the next character
            last_time_step_logits = logits[:, -1, :]  # [batch_size (of one), (last) timestep, (all) logits]
            p = F.softmax(last_time_step_logits, dim=-1)  # logits to probabilities
            idx_next = torch.multinomial(p, num_samples=1)  # sample from the distribution to get the next token
            indices = torch.cat([indices, idx_next], dim=-1)  # append to the sequence
        generated_indices = indices.tolist()[0]  # [0] to remove the batch dim
        if tokenizer is None:
            return generated_indices
        return tokenizer.decode(generated_indices, **tk_kwargs)


def print_model_parameters(model):
    nr_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    nr_non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Number of trainable/non-trainable parameters: {nr_trainable_params:,} / {nr_non_trainable_params:,}")