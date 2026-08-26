import torch
from torch import nn
from torch.nn import functional as F

# Number of distribution parameters predicted per channel. For RGB scale it's 4 parameters: 
# (mu, sigma, pi, lambda). For grayscale scale it would be 3 parameters: (mu, sigma, pi).
_NUM_PARAMS_RGB = 4


def non_shared_get_Kp(K: int, C: int, num_params: int) -> int:
    """Return `Kp`, the number of output channels of the probability head.
    Non-shared means that the K mixture weights (pi_k) are not shared across
    the C image channels.

    Kp = num_params * C * K, where C is the number of image channels and K
    is the number of mixture components.
    """
    return num_params * C * K


def non_shared_get_K(Kp: int, C: int, num_params: int) -> int:
    """Basically the inverse of `non_shared_get_Kp`: recover `K` from `Kp`."""
    return Kp // (num_params * C)


class DiscretizedMixLogisticLoss(nn.Module):
    """Negative log-likelihood under a discretized logistic mixture.

    Inspired by the L3C / PixelCNN++ implementation. Given a target image x (NCHW float)
    and the predicted distribution parameters `l` (N, Kp, H, W), the forward pass returns
    the per-pixel NLL in NCHW layout.
    """

    def __init__(self,
                 x_min: float = 0,
                 x_max: float = 255,
                 n_symbols: int = 256) -> None:
        super().__init__()
        self.x_min = x_min
        self.x_max = x_max
        self.n_symbols = n_symbols

        self._num_params = _NUM_PARAMS_RGB
        # Sigmoid (instead of tanh) for the coefficient activation; matches the
        # SReC variant of the original PixelCNN++ implementation.
        self._nonshared_coeffs_act = torch.sigmoid

        self.bin_width = (x_max - x_min) / (n_symbols - 1)
        # Add a small epsilon to the lower and upper bounds for numerical stability when calculating the CDF.
        self.x_lower_bound = x_min + 0.001
        self.x_upper_bound = x_max - 0.001

    def _cdf_at_centered_offsets(self, offsets: torch.Tensor, inv_stdv: torch.Tensor) -> torch.Tensor:
        """Logistic CDF evaluated at `(value - mean)` offsets given the precomputed 1/sigma.

        `offsets` and `inv_stdv` must be broadcastable. Returns `sigmoid(inv_stdv * offsets)`,
        i.e. the CDF of the logistic distribution shifted/scaled by the supplied parameters.
        """
        return torch.sigmoid(inv_stdv * offsets)

    def log_cdf(self, lo: torch.Tensor, hi: torch.Tensor, means: torch.Tensor, log_scales: torch.Tensor) -> torch.Tensor:
        """Return log(P(lo <= X <= hi)) under a logistic with given params.

        Boundary symbols (lo == x_min or hi == x_max) are treated as open-ended so the total
        probability mass sums to 1. The result is clamped at 1e-6 before taking the log for
        numerical stability.

        The method returns the log of the probability mass on a single discrete bin for
        each of the K components (i.e. output tensor shape is NCKHW).
        """
        centered_lo = lo - means
        centered_hi = hi - means

        # All of the following has shape NCKHW

        # Calculate exp(-log(sigma)) = 1/sigma
        inv_stdv = torch.exp(-log_scales)

        # Sigmoid is, by definition, the CDF of the standard logistic distribution; evaluate
        # it at both bin edges via the shared helper.
        cdf_lo = self._cdf_at_centered_offsets(centered_lo - self.bin_width / 2, inv_stdv)
        lo_cond = (lo >= self.x_lower_bound).float()
        cdf_lo = lo_cond * cdf_lo
        cdf_hi = self._cdf_at_centered_offsets(centered_hi + self.bin_width / 2, inv_stdv)
        hi_cond = (hi <= self.x_upper_bound).float()
        cdf_hi = hi_cond * cdf_hi + (1 - hi_cond)

        cdf_delta = cdf_hi - cdf_lo  # The probability mass on a single discrete bin (the target pixel value)
        log_cdf_delta = torch.log(torch.clamp(cdf_delta, min=1e-6))

        return log_cdf_delta

    def forward(self, x: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
        """Compute the per-pixel NLL.

        Args:
            x: targets, NCHW float in [x_min, x_max].
            l: predicted distribution parameters, N x Kp x H x W.

        Returns:
            NLL of shape NCHW.
        """
        # x: NC1HW, logit_pis: NCKHW, means: NCKHW, log_scales: NCKHW
        x, logit_pis, means, log_scales = self._extract_non_shared(x, l)
        # Same value for lo and hi because that gives us the probability mass on a single discrete bin
        log_probs = self.log_cdf(x, x, means, log_scales)

        log_weights = F.log_softmax(logit_pis, dim=2)  # NCKHW
        log_probs_weighted = log_weights + log_probs  # NCKHW
        nll = -torch.logsumexp(log_probs_weighted, dim=2)  # NCHW (sum over K components)
        return nll

    def expected_entropy(self, x: torch.Tensor, l: torch.Tensor, chunk_size: int = 32) -> torch.Tensor:
        """Per-pixel entropy (in nats) of the predicted discrete distribution.

        The distribution for each channel is conditioned on the true preceding channel
        values via the same lambda coupling used in `forward`, so G is conditioned on R
        and B on R and G.

        Note: this method is computationally very heavy. Running this on CPU is possible, but extremely slow
        (tens of seconds possibly minutes for a large image). Make sure you run this on GPU. If you run out
        of GPU memory, you can try to reduce the chunk size.

        Args:
            x: targets/conditioning, NCHW float in [x_min, x_max].
            l: predicted distribution parameters, N x Kp x H x W.
            chunk_size: number of symbol values processed per iteration. Bounds the peak
                memory of the intermediate `NCKHW * chunk_size` pmf tensor. Pass `self.n_symbols`
                to process the full symbol axis in one shot.
                Why is this important? If the image has shape (C, H, W), then the total number of
                fp32 (4 bytes) numbers in the intermediate tensor are (N*C*K*H*W*chunk_size). If we
                wouldn't chunk, then, with a batch size of 1 and 256 symbols (pixel values) and an image
                size of 1024x1024, the intermediate tensor could be of size (1*3*10*1024*1024*256) * 4 bytes / 1024^3 = 30 GiB.
                Hence we need to be cautious with the chunk size.

        Returns:
            Expected entropy per pixel of shape NCHW.
        """
        _, logit_pis, means, log_scales = self._extract_non_shared(x, l)

        # Add a dimension for broadcasting later.
        means = means.unsqueeze(-1)                            # NCKHW1
        inv_stdv = torch.exp(-log_scales).unsqueeze(-1)        # NCKHW1
        log_w = F.log_softmax(logit_pis, dim=2).unsqueeze(-1)  # NCKHW1

        N, C, _, H, W, _ = means.shape
        chunk = max(1, min(chunk_size, self.n_symbols))

        # Internal bin edges in raw value space (self.n_symbols - 1 of them between consecutive symbols).
        edges_all = self.x_min + self.bin_width * (
            torch.arange(self.n_symbols - 1, device=means.device, dtype=means.dtype) + 0.5
        )

        entropy = torch.zeros(N, C, H, W, device=means.device, dtype=means.dtype)

        for start in range(0, self.n_symbols, chunk):
            end = min(start + chunk, self.n_symbols)
            # Indices into edges_all for the edges bounding symbols [start, end);
            # -1 and self.n_symbols - 1 are out of range and correspond to the open-ended boundaries.
            edges_chunk = edges_all[max(0, start - 1):min(self.n_symbols - 1, end)]

            offsets = edges_chunk - means                                 # NCKHW * chunk_size
            cdf_chunk = self._cdf_at_centered_offsets(offsets, inv_stdv)  # NCKHW * chunk_size

            # Open-ended boundary handling: P(X < x_min) = 0, P(X <= x_max) = 1.
            pieces = []
            if start == 0:
                pieces.append(torch.zeros_like(cdf_chunk[..., :1]))
            pieces.append(cdf_chunk)
            if end == self.n_symbols:
                pieces.append(torch.ones_like(cdf_chunk[..., :1]))
            cdf_chunk = torch.cat(pieces, dim=-1) if len(pieces) > 1 else cdf_chunk

            log_pmf_chunk = (cdf_chunk[..., 1:] - cdf_chunk[..., :-1]).clamp(min=1e-6).log()  # NCKHW * n
            log_marg_chunk = torch.logsumexp(log_w + log_pmf_chunk, dim=2)                    # NCHW * n
            entropy = entropy - (log_marg_chunk.exp() * log_marg_chunk).sum(dim=-1)

        return entropy

    def _extract_non_shared(self, x: torch.Tensor, l: torch.Tensor):
        """Split l into (logit_pi, mu, log_sigma, coeffs) per channel.

        Returns (x_reshaped, logit_probs, means, log_scales) where the means already
        include the lambda-based cross-channel coupling. C is hard-coded to 3 because
        only 3 RGB-style coefficients exist.

        Meaning of "non-shared":
        - Shared (original PixelCNN++): a single set of K mixture weights pi_k is shared across all C=3 channels.
          Means / scales / coefficients are still per-channel, but the mixing proportions are not.
        - Non-shared (L3C and therefore SReC): each channel gets its own independent set of K mixture weights,
          means, and scales. Total parameter count per pixel = num_params * C * K
        """
        N, C, H, W = x.shape
        Kp = l.shape[1]
        K = non_shared_get_K(Kp, C, self._num_params)

        l = l.reshape(N, self._num_params, C, K, H, W)
        logit_pis = l[:, 0, ...]
        means = l[:, 1, ...]
        log_scales = torch.clamp(l[:, 2, ...], min=-7.0)  # Clamp the log-scale parameter to a reasonable value.
        x = x.reshape(N, C, 1, H, W)

        coeffs = self._nonshared_coeffs_act(l[:, 3, ...])  # N3KHW where 3 is for coeffs_g_r, coeffs_b_r, coeffs_b_g
        coeffs_g_r = coeffs[:, 0, ...]  # NKHW
        coeffs_b_r = coeffs[:, 1, ...]  # NKHW
        coeffs_b_g = coeffs[:, 2, ...]  # NKHW
        means = torch.stack((
            means[:, 0, ...],
            means[:, 1, ...] + coeffs_g_r * x[:, 0, ...],
            means[:, 2, ...] + coeffs_b_r * x[:, 0, ...] + coeffs_b_g * x[:, 1, ...]
            ), dim=1)

        means = torch.clamp(means, min=self.x_min, max=self.x_max)  # N3KHW (means for R, G, B channels)
        return x, logit_pis, means, log_scales
