from typing import List, Tuple

import numpy as np
import torch
from torchaudio.functional import edit_distance

from svtr.model.ctc_decoder import CTCDecoder


class NormalizedEditDistance:
    """
    Documentation:
    https://pytorch.org/audio/main/generated/torchaudio.functional.edit_distance.html

    As a side product and to prevent duplicate decoding, this metric also returns
    the text accuracy.
    """

    def __init__(self, decoder: CTCDecoder):
        self.decoder = decoder
        self.avg_ned = -1
        self.avg_acc = -1
        self.n = 0

    def __call__(self,
                 y_pred: torch.Tensor,
                 y_true: torch.Tensor,
                 input_lengths: torch.Tensor,
                 target_lengths: torch.Tensor,
                 *args,
                 **kwargs) -> List[float]:
        y_pred_indices, _ = self.decoder(y_pred, input_lengths, to_text=False)

        ned_batch = [edit_distance(pred, label[:ll]) / ll for pred, label, ll in zip(y_pred_indices, y_true, target_lengths)]
        accuracy_batch = [torch.equal(pred, label[:ll]) for pred, label, ll in zip(y_pred_indices, y_true.to('cpu'), target_lengths)]

        self.update(ned_batch, accuracy_batch)
        return ned_batch

    def update(self, ned_list: List[float], acc_list: List[bool]):
        """Incrementally update the average normalized edit distance.
        """
        nr_samples = len(ned_list)
        new_n = self.n + nr_samples
        # second term: np.sum(ned_list) / new_n = np.mean(ned_list) * (nr_samples / new_n)
        self.avg_ned = self.avg_ned * (self.n / new_n) + np.sum(ned_list) / new_n
        self.avg_acc = self.avg_acc * (self.n / new_n) + np.sum(acc_list) / new_n
        self.n += nr_samples

    def ned_result(self) -> float:
        return self.avg_ned

    def acc_result(self) -> float:
        return self.avg_acc

    def reset(self):
        self.avg_ned = -1
        self.avg_acc = -1
        self.n = 0
