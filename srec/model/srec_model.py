from typing import Generator, List, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from srec.model.srec_blocks import DilatedConvsProbabilityClassifier, EDSRDec, PixelShuffleUpsampler, conv
from srec.model.srec_loss import DiscretizedMixLogisticLoss
from srec.model.srec_outputs import Bits, LogisticMixtureParameters


def tensor_round(x: torch.Tensor) -> torch.Tensor:
    """Round `x` toward the nearest integer with a `-0.001` bias.

    The bias breaks half-integer ties consistently so that the rounded result
    can be re-fed into the network as integer pixel values without sign-flip
    artefacts at exact half-points.
    """
    return torch.round(x - 0.001)


def pad_to_even(x: torch.Tensor) -> torch.Tensor:
    """Replicate-pad `x` along H and W so both dimensions become even.

    Used before `avg_pool2d` to guarantee a clean `2x` downsample even for
    images whose dimensions are odd at the current pyramid level.
    """
    _, _, h, w = x.size()
    pad_right = w % 2 == 1
    pad_bottom = h % 2 == 1
    padding = [0, 1 if pad_right else 0, 0, 1 if pad_bottom else 0]
    return F.pad(x, padding, mode="replicate")


def pad(x: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Replicate-pad `x` on the right / bottom so its size becomes `H x W`."""
    _, _, xH, xW = x.size()
    padding = [0, W - xW, 0, H - xH]
    return F.pad(x, padding, mode="replicate")


def average_downsamples(x: torch.Tensor, scale: int) -> List[torch.Tensor]:
    """Build the image pyramid used by the compressor.

    Returns `scale + 1` tensors where index `0` is the original image and
    index `i` is obtained by `i` successive `(tensor_round -> pad_to_even
    -> avg_pool2d(2))` steps applied to its predecessor. All tensors are
    detached because the pyramid is treated as fixed data, not as a
    differentiable function of the input.
    """
    downsampled = []
    for _ in range(scale):
        downsampled.append(x.detach())
        x = F.avg_pool2d(pad_to_even(tensor_round(x)), 2)
    downsampled.append(x.detach())
    return downsampled


def group_2x2(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split an NCHW tensor into the four pixels of every 2x2 patch.

    Returns the top-left, top-right, bottom-left and bottom-right pixels as
    four `N x C x H/2 x W/2` tensors. Odd dimensions simply drop the last row / column.
    """
    _, _, h, w = x.size()
    x_even_height = x[:, :, 0:h:2, :]
    x_odd_height = x[:, :, 1:h:2, :]
    return (x_even_height[:, :, :, 0:w:2],
            x_even_height[:, :, :, 1:w:2],
            x_odd_height[:, :, :, 0:w:2],
            x_odd_height[:, :, :, 1:w:2])


class Autoregressive2x2Decoder(nn.Module):
    """Super-resolution decoder for one scale of the pyramid.

    Predicts the three first pixels of every 2x2 patch autoregressively using
    a chain of three `EDSRDec` decoders. After each step the feature map is reused
    as additional context for the next pixel, and the pixel-sum constraint (the four
    pixels must sum to `4 * x`, with `x` being the lower-resolution average) is
    propagated to clamp `[lower, upper]` of the logistic-mixture support. The fourth
    pixel is fully determined by the constraint, so no distribution is predicted for it.
    """

    def __init__(self,
                 level: int,
                 n_feats: int,
                 K: int,
                 resblocks: int,
                 expected_entropy: bool = False) -> None:
        super().__init__()
        self.loss_fn = DiscretizedMixLogisticLoss()
        self.level = level  # The level in the image pyramid. The higher the level, the more downsampled the image is.
        self.n_feats = n_feats
        self.expected_entropy = expected_entropy  # Whether to accumulate the expected entropy next to the bit counts.
        # The input channels grow from 3 to 9 as we autoregressively predict the pixels of the 2x2 patch.
        self.rgb_decs = nn.ModuleList([
            EDSRDec(3 * i, n_feats, resblocks=resblocks, tail="conv") for i in range(1, 4)
        ])
        self.mix_logits_prob_clf = nn.ModuleList([
            DilatedConvsProbabilityClassifier(n_feats, C=3, K=K, num_params=self.loss_fn._num_params) for _ in range(1, 4)
        ])
        self.feat_convs = nn.ModuleList([
            conv(n_feats, n_feats, 3) for _ in range(1, 4)
        ])

    def forward_params(self, x_l_1: torch.Tensor, ctx: torch.Tensor) -> Generator[LogisticMixtureParameters, torch.Tensor, torch.Tensor]:
        """
        Generator that yields the distribution parameters for the first 3 pixels in the 2x2 patch.
        The fourth (and last) pixel is fully determined by the constraint, so no distribution is
        predicted for it. Returns the context map once exhausted.
        """
        # x_l_1 is in [0, 255]; pix_sum is the sum of the four 2x2 pixels in x_l.
        pix_sum = x_l_1 * 4  # The sum of the four 2x2 pixels is 4x the pixel value of the downsampled image x_l_1.
        x_l_1_normalized = x_l_1 / 127.5 - 1
        x_l_pixel = torch.tensor([], device=x_l_1.device)
        _, _, h, w = x_l_1.shape

        # Loop over the first 3 pixels in the 2x2 patch and predict their distribution parameters.
        for i, (rgb_dec, clf, feat_conv) in enumerate(
                zip(self.rgb_decs, self.mix_logits_prob_clf, self.feat_convs)):
            # Normalize the pixel value to [-1, 1] and concatenate with the previous pixel values
            # since we are autoregressively predicting the pixels of the 2x2 patch.
            x_l_1_normalized = torch.cat((x_l_1_normalized, x_l_pixel / 127.5 - 1), dim=1)

            # Predict distribution parameters for all 3 color channels at the same time. The R->G->B dependency is created
            # one level down, inside DiscretizedMixLogisticLoss, using the standard PixelCNN++ "coefficients" trick.
            # N x {3, 6, 9} x h x w (note: _, _, h, w = x_l_1.shape) -> N x n_feats x h x w
            z = rgb_dec(x_l_1_normalized, ctx)
            mix_params = clf(z)  # Produces Kp = num_params * C * K logistic-mixture parameters per spatial position.
            # Update the context map for the next pixel.
            ctx = feat_conv(z)

            # Pixel-sum constraint: every remaining pixel still has to leave enough budget
            # for the (3 - i) pixels yet to be predicted, and cannot exceed the running pix_sum.
            # Of course the minimum lower bound is 0 and the maximum upper bound is 255.
            lower = torch.max(pix_sum - (3 - i) * 255, torch.tensor(0., device=x_l_1.device))
            upper = torch.min(pix_sum, torch.tensor(255., device=x_l_1.device))

            # We yield the LogisticMixtureParameters for the current pixel and then expect the
            # next pixel value from the caller through the gen.send() method.
            # x_l_1: the coarse average-pooled image (what we condition on)
            # x_l_pixel: one quadrant of the finer image (what we are encoding)
            x_l_pixel = yield LogisticMixtureParameters(f"lvl_{self.level}_pxl_{i}", mix_params, lower, upper)
            x_l_pixel = pad(x_l_pixel, h, w)  # Pad sothat we can concatenate x_l_pixel with x_l_1_normalized for the next pixel.
            pix_sum = pix_sum - x_l_pixel  # Update the pixel-sum constraint for the next pixel.

        return ctx

    def forward(self, x_l_1: torch.Tensor, x_l: torch.Tensor, ctx: torch.Tensor) -> Tuple[Bits, torch.Tensor]:
        """Encode one pyramid scale and return its bit cost and updated context map.

        Drives the `forward_params` generator to obtain logistic-mixture parameters
        for the first three pixels of every 2x2 patch of `x_l`.
        Saves their NLL into a Bits accumulator, and also computes and saves the rounding-residual bits
        for this scale.
        The fourth pixel of every patch is not coded, since it is fully determined by
        the pixel-sum constraint.

        Args:
            x_l_1: coarser-scale image, shape N x C x H/2 x W/2, float in [0, 255].
                This is the average-pooled version that the decoder conditions on.
            x_l: finer-scale image to encode, shape N x C x H x W, integer-valued
                floats in [0, 255].
            ctx: context feature map from the previous (coarser) scale,
                shape N x n_feats x H/2 x W/2 (0.0 at the coarsest scale).

        Returns:
            bits: Bits accumulator containing the rounding-residual cost and the per-pixel NLL costs
                for this scale.
            ctx: Context map refined by the autoregressive prediction steps at this scale, shape N x n_feats x H x W.
                Passed to the upsampler before being fed into the next finer scale's decoder.
        """
        bits = Bits(expected_entropy=self.expected_entropy)

        # The rounding residual is encoded as a uniform 4-symbol code; the residual only enters
        # through its element count, since a uniform code has a fixed per-symbol cost.
        deltas = x_l_1 - tensor_round(x_l_1)
        bits.add_uniform(f"lvl_{self.level}_rounding", deltas, n_symbols=4)

        _, _, x_h, x_w = x_l_1.shape
        if not isinstance(ctx, float):
            ctx = ctx[..., :x_h, :x_w]

        # Divide pixels of x_l into 2x2 grids. x_l: N 3 H W -> N 4 3 H/2 W/2 (N is batch size)
        x_l_2x2 = group_2x2(x_l)

        gen = self.forward_params(x_l_1, ctx)
        try:
            # Loop over each pixel in the 2x2 patch, run forward_params for that pixel
            # and book the cost in bits for each of those (sub-)pixels into the Bits accumulator.
            for i, x_l_pixel in enumerate(x_l_2x2):
                if i == 0:
                    # We get the first LogisticMixtureParameters for the first pixel. No previous pixel value
                    # to send as it's the first pixel to predict.
                    lm_params = next(gen)
                else:
                    # We send the previous pixel value to the generator and get the LogisticMixtureParameters
                    # for the current pixel.
                    lm_params = gen.send(x_l_2x2[i - 1])
                _, _, h, w = x_l_pixel.shape
                lm_params = LogisticMixtureParameters(
                    name=lm_params.name,
                    dist_params=lm_params.dist_params[..., :h, :w],
                    lower=lm_params.lower[..., :h, :w],
                    upper=lm_params.upper[..., :h, :w]
                )
                bits.add_lm(x_l_pixel, lm_params, self.loss_fn)
        except StopIteration as e:
            # When the generator is exhausted, StopIteration is raised.
            # The return value of the generator is the context map.
            ctx = e.value

        return bits, ctx


class SReC(nn.Module):
    """Hierarchical SReC compressor.

    Re-implementation of the lossless image compressor described in
    "Lossless Image Compression through Super-Resolution" (https://arxiv.org/abs/2004.02872)
    by Jianmin Bao, Dong Chen, Fangde Liu, and Wenhan Luo.

    The model is hierarchical: an image is repeatedly average-pooled down by factors
    of two, producing a pyramid of progressively smaller frames. Each scale is
    reconstructed from the one below by an EDSR-style decoder that predicts the
    parameters of a discretized logistic mixture over the four pixels of every
    2x2 patch (factorised via the PixelCNN++ trick into three conditional distributions).

    Args:
        n_downsamples: Number of pyramid levels under the original image. Must
            be >=0; 0 disables the hierarchical path and codes the image uniformly.
            Default 3 matches SReC's reference config.
        n_feats: Channel width of the EDSR decoders and context maps.
        resblocks: Number of residual blocks per :class:`EDSRDec`.
        K: Number of mixture components per channel in the discretized logistic mixture.
        expected_entropy: Also accumulate the expected entropy of the predicted
            distributions, so that the coding gap against the actual bit counts can be
            measured. This is much slower than the bit counts alone and is not needed
            for training, so it is off by default.
    """

    def __init__(self,
                 n_downsamples: int = 3,
                 n_feats: int = 64,
                 resblocks: int = 3,
                 K: int = 10,
                 expected_entropy: bool = False) -> None:
        super().__init__()
        assert n_downsamples >= 0, n_downsamples

        self.n_downsamples = n_downsamples
        self.n_feats = n_feats
        self.resblocks = resblocks
        self.K = K
        self.expected_entropy = expected_entropy

        # Context upsampler module list for each scale.
        # The first context upsampler is the identity because there is no context yet at the fully compressed image.
        self.ctx_upsamplers = nn.ModuleList([
            nn.Identity(),
            *[PixelShuffleUpsampler(scale=2, n_feats=n_feats) for _ in range(n_downsamples - 1)]
        ] if n_downsamples > 0 else [])

        # Autoregressive2x2Decoder module list for each scale
        self.decs = nn.ModuleList([
            Autoregressive2x2Decoder(level=lvl, n_feats=n_feats, K=K, resblocks=resblocks, expected_entropy=expected_entropy)
            for lvl in range(n_downsamples)[::-1]
        ])

    def forward(self, x: torch.Tensor) -> Bits:
        # Downsampled image pyramid with shapes: [HxW, H/2xW/2, H/4xW/4, ..., H/2^scale x W/2^scale]
        downsampled = average_downsamples(x, self.n_downsamples)

        # Create a Bits accumulator to store the bits for all scales.
        bits = Bits(expected_entropy=self.expected_entropy)
        bits.add_uniform(f"img_lvl_{self.n_downsamples}", tensor_round(downsampled[-1]), n_symbols=256)

        # Initialize the context map for the first scale to 0.
        ctx: torch.Tensor = 0.

        for i in range(self.n_downsamples):
            # Prepare the input tensors for the current scale.
            x_l_1 = downsampled[-i-1]  # x level l+1: H/2xW/2
            y_l = downsampled[-i-2]  # y level l: HxW
            x_l = tensor_round(y_l)  # x level l: HxW

            # Upsample the context map for the next scale.
            ctx = self.ctx_upsamplers[i](ctx)

            # Run the decoder for the current scale and get the bits for the current scale.
            dec_bits, ctx = self.decs[i](x_l_1, x_l, ctx)

            # Add the bits for the current scale to the total bits accumulator.
            bits.update(dec_bits)

        return bits
