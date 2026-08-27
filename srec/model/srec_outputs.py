"""Output types of the SReC compressor: the predicted distribution parameters
and the accumulator that turns them into a bit cost.
"""

from typing import Dict, KeysView, NamedTuple, Union

import torch

from srec.model.srec_loss import DiscretizedMixLogisticLoss


class LogisticMixtureParameters(NamedTuple):
    """Distribution prediction for one of the four pixels of a 2x2 patch.

    Carries the predicted parameters `dist_params` (shape N x Kp x H x W), the
    valid range [lower, upper] derived from the pixel-sum constraint and the
    pyramid scale name used as a logging key.

    The dist_params tensor contains the Kp = num_params * C * K logistic-mixture parameters
    per spatial position; logit_pis, mu, log_sigma and lambda.
    """
    name: str
    dist_params: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor


class Bits:
    """Accumulator for the bits-per-sub-pixel (bpsp) contributions of one pass.

    A single forward of teh SReC model emits a number of independent sub-codes:
    1. The fully downsampled image (or the entire batch of images) is encoded with
       a uniform 256-symbol code. This is the coarsest pyramid level where the fully
       downsampled image is basically stored as a 8-bit image.
    2. The rounding residual is encoded as a uniform 4-symbol code. At every level
       of the pyramid, a rounding residual computed.
    3. For every finer scale, the three pixels of every 2x2 patch are encoded with
       a logistic-mixture code.

    This class stores the negative log-likelihood in bits and the sub-pixel count for
    each sub-code under a string key, so different bpsp views (per code, per scale,
    total over the input) can be queried after the forward.

    Args:
        expected_entropy: Whether to also accumulate the expected entropy of the
            predicted distributions next to their negative log-likelihood. The gap
            between the two is the coding overhead of the model. Computing it is
            far more expensive than the log-likelihood, so it is off by default.
    """

    def __init__(self, expected_entropy: bool = False) -> None:
        self.expected_entropy = expected_entropy

        self.key_to_bits: Dict[str, torch.Tensor] = {}  # Shape (N,) i.e. one value per sample in the batch
        self.key_to_entropy: Dict[str, torch.Tensor] = {}  # Shape (N,) i.e. one value per sample in the batch
        self.key_to_sizes: Dict[str, torch.Tensor] = {}  # Shape (N,) i.e. one value per sample in the batch

    def add_with_size(self,
                      key_to_value: Dict[str, torch.Tensor],
                      key: str,
                      value_per_sample: torch.Tensor,
                      size_per_sample: torch.Tensor) -> None:
        assert key not in key_to_value, f"{key} already exists"
        key_to_value[key] = value_per_sample / torch.log(torch.tensor(2., device=value_per_sample.device))  # Convert nats to bits
        self.key_to_sizes[key] = size_per_sample

    def add(self, key_to_value: Dict[str, torch.Tensor], key: str, value: torch.Tensor) -> None:
        """Add the value to the given dictionary. The value is summed over
        the CHW dimensions.

        Args:
            key_to_value: The dictionary to add the value to, either key_to_bits or key_to_entropy.
            key: The key to add the value to.
            value: The NLL or expected entropy tensor of shape NCHW.
        """
        value_per_sample = value.sum(dim=(1, 2, 3))  # Sum over CHW dimensions
        n, c, h, w = value.shape
        size_per_sample = torch.full((n,), c*h*w, device=value.device)
        self.add_with_size(key_to_value, key, value_per_sample, size_per_sample)

    def add_lm(self, x_l_pixel: torch.Tensor, lm_params: LogisticMixtureParameters, loss_fn: DiscretizedMixLogisticLoss) -> None:
        """Charge the bits needed to encode x_l_pixel under a learned logistic-mixture model.

        Used for the three predicted pixels of every 2x2 patch at every finer scale of the
        pyramid. loss_fn evaluates the per-position negative log-likelihood of x_l_pixel
        under the mixture parameters in lm_params.dist_params; the total is converted from nats
        (natural unit of information, i.e. log base e) to bits (log base 2) and stored
        under lm_params.name.

        Contrasts with add_uniform: here the per-symbol cost depends on the predicted
        distribution and on the actual value of x_l_pixel.
        """
        assert lm_params.dist_params.shape[-2:] == x_l_pixel.shape[-2:], (lm_params.dist_params.shape, x_l_pixel.shape)
        nll = loss_fn(x_l_pixel, lm_params.dist_params)  # NCHW
        self.add(self.key_to_bits, lm_params.name, nll)
        if self.expected_entropy:
            expected_entropy = loss_fn.expected_entropy(x_l_pixel, lm_params.dist_params)  # NCHW
            self.add(self.key_to_entropy, lm_params.name, expected_entropy)

    def add_uniform(self, key: str, x: torch.Tensor, n_symbols: int = 256) -> None:
        """Charge the bits needed to encode x under a flat n_symbols code.

        Used for symbols that are not modelled by the network:
        1. The coarsest pyramid level (whole pixels coded with n_symbols=256).
        2. The rounding residuals at every finer scale (n_symbols=4).
        The cost is the constant size * log2(n_symbols) bits and depends only on x.numel(),
        not on the actual symbol values. A uniform code is its own entropy, so the same
        cost is booked as the expected entropy.

        Note that n_symbols basically describes the size of the symbol alphabet. For encoding
        image pixel values, there are 256 possible values (0-255). For encoding rounding
        residuals, there are 4 possible values (-0.25, 0, 0.25, 0.5).

        Contrast with add_lm: here no learned distribution is involved and the per-symbol
        cost is fixed. This also means that this method is only there to calculate the real
        bpsp metric (otherwise it would be an under-estimated metric). It has no influence
        in the model training (i.e. no gradients are computed).
        """
        n = x.shape[0]
        num_elements = x.numel() // n
        size = torch.full((n,), num_elements, device=x.device)
        nats_under_uniform = torch.log(torch.tensor(float(n_symbols), device=x.device)) * size
        self.add_with_size(self.key_to_bits, key, nats_under_uniform, size)
        if self.expected_entropy:
            self.add_with_size(self.key_to_entropy, key, nats_under_uniform, size)

    def get_keys(self) -> KeysView:
        return self.key_to_bits.keys()

    def _get_self_bpsp(self, key_to_value: Dict[str, torch.Tensor], key: str) -> torch.Tensor:
        return (key_to_value[key] / self.key_to_sizes[key]).mean()

    def _get_total_bpsp(self, key_to_value: Dict[str, torch.Tensor], inp_size: Union[int, torch.Tensor]) -> torch.Tensor:
        _example = next(iter(key_to_value.values()))  # Fast way to extract the first value from the dictionary
        total = torch.zeros(_example.shape[0], device=_example.device)
        for key in key_to_value:
            total = total + key_to_value[key]
        return total / inp_size

    def _assert_expected_entropy(self) -> None:
        assert self.expected_entropy, "The expected entropy is only available when Bits is created with expected_entropy=True"

    def get_self_bpsp(self, key: str) -> torch.Tensor:
        """Get mean bpsp of the batch for the given key."""
        return self._get_self_bpsp(self.key_to_bits, key)

    def get_self_bpsp_per_layer(self) -> Dict[str, torch.Tensor]:
        return {key: self.get_self_bpsp(key) for key in self.get_keys()}

    def get_self_expected_entropy_per_layer(self) -> Dict[str, torch.Tensor]:
        """Per-key counterpart of get_self_bpsp_per_layer, in bits per sub-pixel."""
        self._assert_expected_entropy()
        return {key: self._get_self_bpsp(self.key_to_entropy, key) for key in self.get_keys()}

    def get_bpsp(self, inp_size: Union[int, torch.Tensor]) -> torch.Tensor:
        """
        inp_size is the number of sub-pixels in the input image. I.e. in an RGB image
        there's 3*H*W sub-pixels.

        It returns the total bits-per-sub-pixel (bpsp) per sample. This is also the
        value to use for the loss (i.e. the bpsp to minimize) when training the model.
        """
        return self._get_total_bpsp(self.key_to_bits, inp_size)

    def get_expected_entropy_bpsp(self, inp_size: Union[int, torch.Tensor]) -> torch.Tensor:
        """Total expected entropy per sample in bits per sub-pixel.

        This is the cost the input would incur under the predicted distributions if the
        symbols were drawn from those distributions. It is therefore a lower bound on
        get_bpsp, and the difference between the two is the coding gap of the model.
        """
        self._assert_expected_entropy()
        return self._get_total_bpsp(self.key_to_entropy, inp_size)

    def update(self, other: "Bits") -> "Bits":
        """Merge `other` into `self`; the two key-sets must be disjoint."""
        assert len(self.get_keys() & other.get_keys()) == 0, f"{self.get_keys()} and {other.get_keys()} intersect."
        assert self.expected_entropy == other.expected_entropy, "Cannot merge accumulators with different collection modes."
        self.key_to_bits.update(other.key_to_bits)
        self.key_to_entropy.update(other.key_to_entropy)
        self.key_to_sizes.update(other.key_to_sizes)
        return self

    def __repr__(self) -> str:
        return f"{self.__class__}: {({k: v.item() for k, v in self.get_self_bpsp_per_layer().items()})}"
