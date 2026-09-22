"""Lossless image codec built on top of a trained `SReC` model.

The model predicts a distribution for every sub-pixel it does not store verbatim; a range
coder turns those distributions into an actual byte stream whose length approaches the
negative log-likelihood that `Bits` reports during training.

Encoding and decoding walk the pyramid in exactly the order `SReC.forward` charges bits for
it, and both directions share `_traverse`, so the two cannot drift apart:

1. The coarsest level is written verbatim under a flat 256 symbol code.
2. For every finer level, the rounding residual of the coarser level is written under a flat
   4 symbol code. The decoders condition on the unrounded coarser image, of which the stream
   so far only carries the rounded form, so the residual is what makes that image exact again.
3. The first three pixels of every 2x2 patch are written under the predicted logistic
   mixtures, one colour channel after the other because the mean of a channel depends on the
   values of the channels before it. The fourth pixel is never written: it follows from the
   four pixels of a patch averaging to the coarser pixel.

The stream only decodes on a machine that reproduces the model's floating point output
bit for bit. The same weights, the same device and the same build of torch are required;
the header carries a fingerprint of the weights so that a mismatch is reported rather than
silently decoded into noise.
"""

import hashlib
import struct
from typing import List, Optional, Tuple

import constriction
import numpy as np
import torch

from srec.model.srec_loss import DiscretizedMixLogisticLoss
from srec.model.srec_model import SReC, average_downsamples, group_2x2, tensor_round

_MAGIC = b"SREC"
_FORMAT_VERSION = 1
_DIGEST_BYTES = 8
# magic, format version, number of pyramid levels, channels, height, width, weight fingerprint.
_HEADER = struct.Struct(f"<4sBBBII{_DIGEST_BYTES}s")

# Rounding residual symbols: -1/4, 0, 1/4, 1/2.
_N_ROUNDING_SYMBOLS = 4
# The coarsest level is stored as a plain 8-bit image.
_N_PIXEL_SYMBOLS = 256

# Model family shared by all learned symbols: one row of probabilities per coded sub-pixel.
_CATEGORICAL = constriction.stream.model.Categorical(perfect=False)


def _uniform_model(n_symbols: int) -> constriction.stream.model.Categorical:
    """Flat distribution over `n_symbols` values, costing log2(n_symbols) bits per symbol."""
    return constriction.stream.model.Categorical(
        np.full(n_symbols, 1.0 / n_symbols, dtype=np.float32), perfect=False
    )


class _StreamEncoder:
    """Writes known symbols into a range coded stream.

    `code` and `code_uniform` return the symbols they were given so that the caller can feed
    them back into the model in the same way the decoder feeds back what it just read.
    """

    def __init__(self) -> None:
        self._coder = constriction.stream.queue.RangeEncoder()

    def code(self, pmf: np.ndarray, symbols: Optional[np.ndarray]) -> np.ndarray:
        self._coder.encode(symbols, _CATEGORICAL, pmf)
        return symbols

    def code_uniform(self, n_symbols: int, count: int, symbols: Optional[np.ndarray]) -> np.ndarray:
        self._coder.encode(symbols, _uniform_model(n_symbols))
        return symbols

    def payload(self) -> bytes:
        """The stream as little-endian unsigned 32-bit integers, independent of the host byte order."""
        return self._coder.get_compressed().astype("<u4").tobytes()


class _StreamDecoder:
    """Reads symbols back out of a range coded stream.

    Mirrors `_StreamEncoder`; the `symbols` argument is the value the encoder wrote and is
    unavailable here, so it is ignored.
    """

    def __init__(self, payload: bytes) -> None:
        words = np.frombuffer(payload, dtype="<u4").astype(np.uint32)
        self._coder = constriction.stream.queue.RangeDecoder(words)

    def code(self, pmf: np.ndarray, symbols: Optional[np.ndarray]) -> np.ndarray:
        return self._coder.decode(_CATEGORICAL, pmf)

    def code_uniform(self, n_symbols: int, count: int, symbols: Optional[np.ndarray]) -> np.ndarray:
        return self._coder.decode(_uniform_model(n_symbols), count)


def weights_digest(model: torch.nn.Module) -> bytes:
    """Fingerprint of the model weights, recorded in the header of every stream.
    For example the model can be a `SReC` model.
    """
    digest = hashlib.blake2b(digest_size=_DIGEST_BYTES)
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.digest()


def _pyramid_shapes(height: int, width: int, n_downsamples: int) -> List[Tuple[int, int]]:
    """Spatial size of every pyramid level, index 0 being the original image.

    Mirrors `pad_to_even` followed by `avg_pool2d`, which rounds an odd size up.
    """
    shapes = [(height, width)]
    for _ in range(n_downsamples):
        h, w = shapes[-1]
        shapes.append(((h + 1) // 2, (w + 1) // 2))
    return shapes


def _quadrant_shapes(height: int, width: int) -> List[Tuple[int, int]]:
    """Sizes of the four pixels of every 2x2 patch of a `height x width` image.

    In the order `group_2x2` produces them. An odd size leaves the last row or column without
    a partner, so the quadrants that would fall outside the image are one shorter.
    """
    up_h, up_w = (height + 1) // 2, (width + 1) // 2
    down_h, down_w = height // 2, width // 2
    return [(up_h, up_w), (up_h, down_w), (down_h, up_w), (down_h, down_w)]


def _residual_symbols(x: torch.Tensor) -> np.ndarray:
    """Rounding residual of `x` as indices into the four quarter steps."""
    quarter_steps = (x - tensor_round(x)) * 4
    if not torch.equal(quarter_steps, quarter_steps.round()):
        raise ValueError("Pyramid level is not a multiple of a quarter; the stream cannot represent it.")
    return torch.remainder(quarter_steps, _N_ROUNDING_SYMBOLS).reshape(-1).to(torch.int32).cpu().numpy()


def _code_pixel(coder,
                loss_fn: DiscretizedMixLogisticLoss,
                dist_params: torch.Tensor,
                true_pixel: Optional[torch.Tensor]) -> torch.Tensor:
    """Code the three colour channels of one quadrant under its predicted mixture.

    The channels go one after the other because the mean of a channel is coupled to the values
    of the channels before it, so a channel can only be modelled once its predecessors are known.

    Args:
        coder: `_StreamEncoder` or `_StreamDecoder`.
        loss_fn: the mixture the `dist_params` parameterise.
        dist_params: predicted distribution parameters, 1 x Kp x h x w.
        true_pixel: the quadrant to encode, 1 x 3 x h x w, or None when decoding.

    Returns:
        The coded quadrant, 1 x 3 x h x w.
    """
    _, _, h, w = dist_params.shape
    pixel = torch.zeros(1, 3, h, w, device=dist_params.device, dtype=dist_params.dtype)
    for channel in range(3):
        pmf = loss_fn.symbol_pmf(pixel, dist_params, channel)[0]  # h x w x n_symbols
        pmf = pmf.reshape(-1, pmf.shape[-1]).to(torch.float32).cpu().numpy()
        pmf = pmf / pmf.sum(axis=1, keepdims=True)  # The range coder needs normalised rows.

        symbols = None if true_pixel is None else true_pixel[0, channel].reshape(-1).to(torch.int32).cpu().numpy()
        symbols = coder.code(pmf, symbols)
        pixel[0, channel] = torch.as_tensor(symbols, dtype=pixel.dtype, device=pixel.device).reshape(h, w)
    return pixel


def _merge_2x2(quadrants: List[torch.Tensor], x_l_1: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Reassemble a level from the three coded pixels of every 2x2 patch.

    The fourth pixel is recovered from the pixel-sum constraint instead of being read from the
    stream. The constraint only holds where the patch is complete, which is the top-left
    `height // 2 x width // 2` block of patches; every pixel outside it is coded explicitly.
    """
    p0, p1, p2 = quadrants
    full_h, full_w = height // 2, width // 2
    p3 = 4 * x_l_1[..., :full_h, :full_w] - p0[..., :full_h, :full_w] - p1[..., :full_h, :full_w] - p2[..., :full_h, :full_w]

    x = torch.empty(1, 3, height, width, device=p0.device, dtype=p0.dtype)
    x[..., 0::2, 0::2] = p0
    x[..., 0::2, 1::2] = p1
    x[..., 1::2, 0::2] = p2
    x[..., 1::2, 1::2] = p3
    return x


def _traverse(srec: SReC, coder, height: int, width: int, image: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Walk the pyramid from the coarsest level up, coding every symbol on the way.

    Encoding passes `image` and discards the return value; decoding passes None and keeps it.
    Both directions issue the same model calls on the same tensors in the same order, which is
    what keeps the streams in sync.

    Args:
        srec: the model whose predictions drive the coder.
        coder: `_StreamEncoder` or `_StreamDecoder`.
        height: height of the original image.
        width: width of the original image.
        image: the image to encode, 1 x 3 x H x W, or None when decoding.

    Returns:
        The original image, 1 x 3 x H x W of integer valued floats.
    """
    device = next(srec.parameters()).device
    shapes = _pyramid_shapes(height, width, srec.n_downsamples)
    pyramid = None if image is None else average_downsamples(image, srec.n_downsamples)

    coarsest_h, coarsest_w = shapes[-1]
    symbols = coder.code_uniform(
        _N_PIXEL_SYMBOLS,
        3 * coarsest_h * coarsest_w,
        None if pyramid is None else tensor_round(pyramid[-1]).reshape(-1).to(torch.int32).cpu().numpy(),
    )
    x_rounded = torch.as_tensor(symbols, dtype=torch.float32, device=device).reshape(1, 3, coarsest_h, coarsest_w)

    ctx: torch.Tensor = 0.
    for i in range(srec.n_downsamples):
        coarse_level = srec.n_downsamples - i
        coarse_h, coarse_w = shapes[coarse_level]
        fine_h, fine_w = shapes[coarse_level - 1]

        symbols = coder.code_uniform(
            _N_ROUNDING_SYMBOLS,
            3 * coarse_h * coarse_w,
            None if pyramid is None else _residual_symbols(pyramid[coarse_level]),
        )
        residual = torch.as_tensor(symbols, dtype=torch.float32, device=device).reshape(1, 3, coarse_h, coarse_w)
        x_l_1 = x_rounded + ((residual + 1) % _N_ROUNDING_SYMBOLS - 1) / 4

        ctx = srec.ctx_upsamplers[i](ctx)
        if not isinstance(ctx, float):
            ctx = ctx[..., :coarse_h, :coarse_w]

        dec = srec.decs[i]
        true_quadrants = None if pyramid is None else group_2x2(tensor_round(pyramid[coarse_level - 1]))
        gen = dec.forward_params(x_l_1, ctx)
        quadrants: List[torch.Tensor] = []
        for j, (quadrant_h, quadrant_w) in enumerate(_quadrant_shapes(fine_h, fine_w)[:3]):
            lm_params = next(gen) if j == 0 else gen.send(quadrants[j - 1])
            quadrants.append(_code_pixel(
                coder,
                dec.loss_fn,
                lm_params.dist_params[..., :quadrant_h, :quadrant_w],
                None if true_quadrants is None else true_quadrants[j],
            ))
        try:
            # Feeding the third pixel back exhausts the generator, which returns the context map.
            gen.send(quadrants[2])
        except StopIteration as stop:
            ctx = stop.value

        x_rounded = _merge_2x2(quadrants, x_l_1, fine_h, fine_w)

    return x_rounded


def _as_batch(image: torch.Tensor) -> torch.Tensor:
    """Validate an image and give it a leading batch dimension of one."""
    if image.dim() == 3:
        image = image.unsqueeze(0)
    if image.dim() != 4 or image.shape[0] != 1 or image.shape[1] != 3:
        raise ValueError(f"Expected a single RGB image of shape (1, 3, H, W), got {tuple(image.shape)}.")
    image = image.float()
    if not torch.equal(image, image.round()) or image.min() < 0 or image.max() > 255:
        raise ValueError("Expected integer valued pixels in the 0-255 range.")
    return image


def encode(srec: SReC, image: torch.Tensor) -> bytes:
    """Compress a single RGB image into a self-describing byte stream.

    Args:
        srec: trained model; its weights are part of the format and are needed to decode.
        image: 3 x H x W or 1 x 3 x H x W of integer valued floats in the 0-255 range.
    """
    image = _as_batch(image).to(next(srec.parameters()).device)
    _, channels, height, width = image.shape

    srec.eval()
    coder = _StreamEncoder()
    with torch.no_grad():
        _traverse(srec, coder, height, width, image)

    # Header fields, little-endian: 4s BBB II 8s.
    # | Format | Meaning              | Value                    |
    # |--------|----------------------|--------------------------|
    # | 4s     | magic                | b"SREC"                  |
    # | B      | format version       | 1                        |
    # | B      | pyramid depth        | srec.n_downsamples       |
    # | B      | channel count        | channels                 |
    # | I      | height               | height                   |
    # | I      | width                | width                    |
    # | 8s     | weight fingerprint   | weights_digest(srec)     |
    header = _HEADER.pack(
        _MAGIC, _FORMAT_VERSION, srec.n_downsamples, channels, height, width, weights_digest(srec)
    )
    return header + coder.payload()


def decode(srec: SReC, stream: bytes) -> torch.Tensor:
    """Restore the image a stream produced by `encode` was made from.

    Args:
        srec: the same model, with the same weights, that produced the stream.
        stream: the bytes returned by `encode`.

    Returns:
        1 x 3 x H x W of integer valued floats in the 0-255 range.
    """
    magic, version, n_downsamples, channels, height, width, digest = _HEADER.unpack_from(stream)
    if magic != _MAGIC:
        raise ValueError("Not an SReC stream.")
    if version != _FORMAT_VERSION:
        raise ValueError(f"Stream format version {version} is not supported, expected {_FORMAT_VERSION}.")
    if n_downsamples != srec.n_downsamples or channels != 3:
        raise ValueError(f"Stream was written by a model with n_downsamples={n_downsamples}, "
                         f"channels={channels}, which does not match the given model.")
    if digest != weights_digest(srec):
        raise ValueError("Stream was written with different model weights and cannot be decoded.")

    srec.eval()
    coder = _StreamDecoder(stream[_HEADER.size:])
    with torch.no_grad():
        return _traverse(srec, coder, height, width)


def is_stream(path: str) -> bool:
    """Whether the file at `path` carries the marker every stream starts with.

    Lets a caller tell a compressed stream from an image without relying on the file name.
    """
    with open(path, "rb") as f:
        return f.read(len(_MAGIC)) == _MAGIC


def save(srec: SReC, image: torch.Tensor, path: str) -> int:
    """Compress `image` to `path` and return the number of bytes written."""
    stream = encode(srec, image)
    with open(path, "wb") as f:
        f.write(stream)
    return len(stream)


def load(srec: SReC, path: str) -> torch.Tensor:
    """Restore the image stored in the stream at `path`."""
    with open(path, "rb") as f:
        return decode(srec, f.read())
