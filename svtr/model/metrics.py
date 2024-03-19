import numpy as np
import torch
from torchaudio.functional import edit_distance


class NormalizedEditDistance:
    """
    Documentation:
    https://pytorch.org/audio/main/generated/torchaudio.functional.edit_distance.html

    As a side product and to prevent duplicate decoding, this metric also returns
    the text accuracy.
    """

    def __init__(self, decoder):
        """
        Args:
            decoder (CTCDecoder):
        """
        self.decoder = decoder
        self.avg_ned = -1
        self.avg_acc = -1
        self.n = 0

    def __call__(self, y_pred, y_true, *args, **kwargs):
        y_pred_indices, _ = self.decoder(y_pred, to_text=False)
        ned_batch = [edit_distance(pred, label) / len(label) for pred, label in zip(y_pred_indices, y_true)]
        accuracy_batch = [torch.equal(pred, label) for pred, label in zip(y_pred_indices, y_true.to('cpu'))]
        self.update(ned_batch, accuracy_batch)
        return ned_batch

    def update(self, ned_list, acc_list):
        """Incrementally update the average normalized edit distance.
        """
        nr_samples = len(ned_list)
        new_n = self.n + nr_samples
        # second term: np.sum(ned_list) / new_n = np.mean(ned_list) * (nr_samples / new_n)
        self.avg_ned = self.avg_ned * (self.n / new_n) + np.sum(ned_list) / new_n
        self.avg_acc = self.avg_acc * (self.n / new_n) + np.sum(acc_list) / new_n
        self.n += nr_samples

    def ned_result(self):
        return self.avg_ned

    def acc_result(self):
        return self.avg_acc

    def reset(self):
        self.avg_ned = -1
        self.avg_acc = -1
        self.n = 0
