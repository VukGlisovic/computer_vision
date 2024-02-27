

class CharacterTokenizer:
    """Tokenizes individual characters into digits (indices). You only
    need to provide the vocabulary (a list of characters) to this class.
    """

    def __init__(self, vocab):
        super().__init__()
        # creating mapping functions for idx to character and vice versa
        self.idx_to_ch = {i: ch for i, ch in enumerate(vocab)}
        self.ch_to_idx = {ch: i for i, ch in self.idx_to_ch.items()}

    def encode(self, string):
        """ Encodes characters into indices. """
        return [self.ch_to_idx[ch] for ch in string]

    def decode(self, indices):
        """ Decodes indices into characters. """
        return ''.join([self.idx_to_ch[i] for i in indices])
