"""Lightning wrapper around the SReC compressor."""

from typing import Any, Dict, List, Tuple

import lightning as L
import torch

from srec.model.srec_model import SReC


class SrecLightningModule(L.LightningModule):
    """Trains a `SReC` model to minimize the bits-per-sub-pixel (bpsp) of its input.

    The compressor returns a `Bits` accumulator rather than a tensor; the loss is the
    total bpsp averaged over the batch. The per-code bpsp contributions (one uniform
    code for the coarsest scale, one rounding-residual code and three logistic-mixture
    codes per finer scale) are logged separately to show where the bit budget goes.

    Args:
        n_downsamples: Number of pyramid levels under the original image.
        n_feats: Channel width of the EDSR decoders and context maps.
        resblocks: Number of residual blocks per EDSR decoder.
        K: Number of mixture components per channel in the discretized logistic mixture.
        expected_entropy: Also log the expected entropy of the predicted distributions and
            the coding gap against the bits actually spent. This costs one pass over all 256
            symbol values per predicted pixel, which slows a step down by roughly two orders
            of magnitude, so it is meant for analysis runs rather than for training.
        learning_rate: Initial AdamW learning rate.
        lr_patience: Number of epochs without an improvement of the validation bpsp
            that are tolerated before the learning rate is decayed.
        lr_factor: Multiplicative factor of the learning rate decay.
    """

    def __init__(self,
                 n_downsamples: int = 3,
                 n_feats: int = 64,
                 resblocks: int = 3,
                 K: int = 10,
                 expected_entropy: bool = False,
                 learning_rate: float = 1e-4,
                 lr_patience: int = 3,
                 lr_factor: float = 0.1) -> None:
        super().__init__()
        self.save_hyperparameters()  # Saves the input arguments to the init of this class
        self.srec = SReC(
            n_downsamples=n_downsamples, 
            n_feats=n_feats, 
            resblocks=resblocks, 
            K=K, 
            expected_entropy=expected_entropy
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.srec(x)

    def _step(self, images: torch.Tensor, stage: str) -> torch.Tensor:
        bits = self.srec(images)
        # The number of sub-pixels of a single image, i.e. C * H * W.
        bpsp = bits.get_bpsp(images[0].numel()).mean()

        on_step = stage == "train"
        self.log(f"{stage}/bpsp", bpsp, prog_bar=True, on_step=on_step, on_epoch=True)
        for key, key_bpsp in bits.get_self_bpsp_per_layer().items():
            self.log(f"{stage}/bpsp_{key}", key_bpsp, on_step=False, on_epoch=True)

        if self.hparams.expected_entropy:
            entropy = bits.get_expected_entropy_bpsp(images[0].numel()).mean()
            self.log(f"{stage}/expected_entropy", entropy, on_step=on_step, on_epoch=True)
            # How many bits per sub-pixel are spent above the entropy of the predicted distributions.
            self.log(f"{stage}/coding_gap", bpsp - entropy, on_step=on_step, on_epoch=True)
            for key, key_entropy in bits.get_self_expected_entropy_per_layer().items():
                self.log(f"{stage}/expected_entropy_{key}", key_entropy, on_step=False, on_epoch=True)
        return bpsp

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")

    def configure_optimizers(self) -> Tuple[List[Any], List[Dict[str, Any]]]:
        """The learning rate decays once the validation bpsp stops improving, which requires
        a validation dataloader to be passed to `fit` so that the monitored metric exists.
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=self.hparams.lr_factor, patience=self.hparams.lr_patience
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch", "monitor": "val/bpsp"}]
