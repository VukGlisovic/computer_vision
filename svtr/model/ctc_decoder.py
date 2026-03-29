from typing import List, Tuple

import torch
from torchaudio.models.decoder import ctc_decoder


class CTCDecoder:

    def __init__(self, vocab: List[str], beam_size: int = 1, blank_token: str = '<BLK>'):
        self.decoder = self.create_ctc_decoder(vocab, beam_size, blank_token)

    def __call__(self, log_softmax: torch.Tensor, lengths: torch.Tensor, to_text: bool = True, *args, **kwargs) -> Tuple[List[List[int]], List[float]]:
        # Get decoding hypotheses
        batch_hypotheses = self.decoder(log_softmax.to('cpu'), lengths=lengths.to('cpu'))  # List[List[CTCHypothesis]]
        # Transcript for a lexicon free decoder, splitting by blank token
        batch_indices = [h[0].tokens for h in batch_hypotheses]
        batch_scores = [h[0].score for h in batch_hypotheses]
        result = batch_indices
        if to_text:
            batch_tokens = [self.decoder.idxs_to_tokens(indices) for indices in batch_indices]
            transcripts = ["".join(tokens) for tokens in batch_tokens]
            result = transcripts
        return result, batch_scores

    @staticmethod
    def create_ctc_decoder(vocab: List[str], beam_size: int = 1, blank_token: str = '<BLK>'):
        """
        Documentation:
        https://pytorch.org/audio/main/generated/torchaudio.models.decoder.ctc_decoder.html

        Note that beam_size=1 is basically greedy decoding.

        Returns:
            ctc_decoder
        """
        decoder = ctc_decoder(
            lexicon=None,
            tokens=vocab,
            lm=None,
            nbest=1,
            beam_size=beam_size,
            beam_threshold=50,
            blank_token=blank_token,
            sil_token=blank_token
        )
        return decoder
