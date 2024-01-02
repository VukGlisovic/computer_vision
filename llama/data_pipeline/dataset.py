import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):

    def __init__(self, text_data, tokenizer, context_window, device, dtype=torch.int8):
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

        encoded_text = tokenizer.encode(self.text_data)
        if hasattr(encoded_text, 'ids'):
            encoded_text = encoded_text.ids
        self.data_tokenized = torch.tensor(encoded_text, dtype=dtype).to(device)

    def __len__(self):
        return len(self.data_tokenized) - self.context_window - 1

    def __getitem__(self, i):
        """Create input sequence (x) and a corresponding target sequence (y)
        which is x shifted to the right.
        """
        x = self.data_tokenized[i: i + self.context_window].long()
        y = self.data_tokenized[i + 1: i + self.context_window + 1].long()
        return x, y
