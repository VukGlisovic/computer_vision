from torchaudio.models.decoder import ctc_decoder


class CTCDecoder:

    def __init__(self, vocab, beam_size=1, blank_token='<BLK>'):
        self.decoder = self.create_ctc_decoder(vocab, beam_size, blank_token)

    def __call__(self, logits, to_text=True, *args, **kwargs):
        # get decoding hypotheses
        batch_hypotheses = self.decoder(logits.to('cpu'))  # List[List[CTCHypothesis]]
        # transcript for a lexicon free decoder, splitting by blank token
        batch_indices = [h[0].tokens for h in batch_hypotheses]
        batch_scores = [h[0].score for h in batch_hypotheses]
        result = batch_indices
        if to_text:
            batch_tokens = [self.decoder.idxs_to_tokens(indices) for indices in batch_indices]
            transcripts = ["".join(tokens) for tokens in batch_tokens]
            result = transcripts
        return result, batch_scores

    @staticmethod
    def create_ctc_decoder(vocab, beam_size=50, blank_token='<BLK>'):
        """
        Documentation:
        https://pytorch.org/audio/main/generated/torchaudio.models.decoder.ctc_decoder.html

        Args:
            vocab (list[str]):
            beam_size (int):
            blank_token (str):

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
