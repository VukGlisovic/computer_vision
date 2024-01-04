from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):

    def __init__(self, text_data, tokenizer, context_window, device, dtype=torch.int8, verbose=1):
        """
        Args:
            text_data (str): one big string with all text concatenated.
            tokenizer (Tokenizer):
            context_window (int):
            device (str): whether to have data on CPU or GPU
            dtype (dtype): torch data type for encoded token ids
        """
        self.text_data = text_data
        self.tokenizer = tokenizer
        self.context_window = context_window
        self.device = device

        text_iter = self.text_data
        if verbose:
            text_iter = tqdm(text_iter, desc="Tokenizing text")
        encoded_texts = [tokenizer.encode(text) for text in text_iter]
        if hasattr(encoded_texts[0], 'ids'):
            encoded_texts = [enc_text.ids for enc_text in encoded_texts]
        self.data_tokenized = [torch.tensor(enc_text, dtype=dtype) for enc_text in encoded_texts]
        # prepend 0 to be able to easily obtain an offset in __get_item__
        text_lengths = [0] + [len(ids) - self.context_window - 1 for ids in self.data_tokenized]
        self.data_tokenized_cum_len = np.cumsum(text_lengths)

    def __len__(self):
        return self.data_tokenized_cum_len[-1]

    def __getitem__(self, i):
        """Create input sequence (x) and a corresponding target sequence (y)
        which is x shifted to the right.
        """
        text_idx = max(np.argmax(self.data_tokenized_cum_len >= i) - 1, 0)  # make sure text_idx=-1 is not possible
        offset = self.data_tokenized_cum_len[text_idx]
        _i = i - offset
        x = self.data_tokenized[text_idx][_i: _i + self.context_window].long()
        y = self.data_tokenized[text_idx][_i + 1: _i + self.context_window + 1].long()
        return x.to(self.device), y.to(self.device)
