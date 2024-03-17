import numpy as np
from torchaudio.functional import edit_distance


class NormalizedEditDistance:
    """
    Documentation:
    https://pytorch.org/audio/main/generated/torchaudio.functional.edit_distance.html
    """

    def __init__(self, decoder):
        """
        Args:
            decoder (CTCDecoder):
        """
        self.decoder = decoder
        self.ned_list = []

    def __call__(self, y_pred, y_true, *args, **kwargs):
        y_pred_indices, _ = self.decoder(y_pred, to_text=False)
        ned = [edit_distance(pred, label) / len(label) for pred, label in zip(y_pred_indices, y_true)]
        self.ned_list += ned
        return ned

    def compute(self):
        return np.mean(self.ned_list)

    def reset(self):
        self.ned_list = []
