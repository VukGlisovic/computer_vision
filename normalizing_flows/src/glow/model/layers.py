from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from scipy import linalg as la

from normalizing_flows.src.realnvp.model.layers import Rescale
from normalizing_flows.src.realnvp.model.layer_utils import weight_norm


class ActNorm(nn.Module):

	def __init__(self, n_channels: int):
		super().__init__()
		self.n_channels = n_channels

		self.loc = nn.Parameter(torch.zeros(1, n_channels, 1, 1))
		self.scale = nn.Parameter(torch.ones(1, n_channels, 1, 1))

		self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))

	def initialize(self, x: torch.Tensor) -> None:
		# input shape: (batch_size, channels, height, width)
		with torch.no_grad():
			# Convert to (channels, batch_size, height, width) and flatten to (channels, batch_size * height * width)
			flattened = x.permute(1, 0, 2, 3).contiguous().view(self.n_channels, -1)
			# Calculate mean and std per channel
			loc = flattened.mean(dim=1).view(1, self.n_channels, 1, 1)
			scale = flattened.std(dim=1).view(1, self.n_channels, 1, 1)

			self.loc.data.copy_(loc)
			self.scale.data.copy_(1 / (scale + 1e-6))

	def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		_, _, h, w = x.shape

		if self.initialized.item() == 0:
			self.initialize(x)
			self.initialized.fill_(1)

		log_abs = torch.log(torch.abs(self.scale))

		# Multiply by height and width to get logdet for each pixel
		logdet = h * w * torch.sum(log_abs)

		return self.scale * (x - self.loc), logdet

	@torch.no_grad()
	def inverse(self, z: torch.Tensor) -> torch.Tensor:
		return z / self.scale + self.loc


class InvertibleConv2d(nn.Module):
	"""
    This class uses LU decomposition to create the weights of the convolution.
    """

	def __init__(self, in_channels: int):
		super().__init__()
		self.in_channels = in_channels

		weight = np.random.randn(in_channels, in_channels)
		# Apply QR decomposition to get an orthonormal matrix q (and an upper triangular matrix)
		# q is orthonormal, so it's basically a rotation
		q, _ = la.qr(weight)
		# Apply LU decomposition to get a lower and upper triangular matrix and a permutation matrix
		# By applying this decomposition, we can very easily calculate the determinant as described in section 3.2 to the Glow paper
		w_p, w_l, w_u = la.lu(q.astype(np.float32))
		# Create a diagonal matrix from the upper triangular matrix
		w_s = np.diag(w_u).copy()
		# Remove the diagonal from the upper triangular matrix
		w_u = np.triu(w_u, 1).copy()
		# Create masks for the upper and lower triangular matrices
		u_mask = np.triu(np.ones_like(w_u), 1).copy()
		l_mask = u_mask.T.copy()

		# The .copy() operation creates a new, writable array with the same data, which is what PyTorch expects when creating tensors
		w_p = torch.from_numpy(w_p.copy())
		w_l = torch.from_numpy(w_l.copy())
		w_s = torch.from_numpy(w_s.copy())
		w_u = torch.from_numpy(w_u.copy())

		self.register_buffer("l_mask", torch.from_numpy(l_mask))
		self.register_buffer("l_eye", torch.eye(l_mask.shape[0]))
		self.register_buffer("u_mask", torch.from_numpy(u_mask))
		self.register_buffer("s_sign", torch.sign(w_s))
		# As described in the Glow paper, we keep the permutation matrix fixed, while we optimize the rest
		self.register_buffer("w_p", w_p)
		self.w_l = nn.Parameter(w_l)
		self.w_s = nn.Parameter(torch.log(torch.abs(w_s)))
		self.w_u = nn.Parameter(w_u)

	def get_weight(self) -> torch.Tensor:
		w_l = self.w_l * self.l_mask + self.l_eye
		w_u = (self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s))
		weight = self.w_p @ w_l @ w_u
		return weight

	def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		_, _, h, w = x.shape

		weight = self.get_weight()
		weight = weight.unsqueeze(2).unsqueeze(3)

		out = F.conv2d(x, weight)
		logdet = h * w * torch.sum(self.w_s)

		return out, logdet

	@torch.no_grad()
	def inverse(self, z: torch.Tensor) -> torch.Tensor:
		weight = self.get_weight()
		weight = weight.inverse().unsqueeze(2).unsqueeze(3)

		return F.conv2d(z, weight)


class Conv2dZeroInit(nn.Conv2d):

	def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: str = 'same'):
		super().__init__(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
		self.weight.data.zero_()
		self.bias.data.zero_()


class AffineCoupling(nn.Module):

	def __init__(self, in_channels: int, hidden_channels: int = 512):
		super().__init__()
		self.in_channels = in_channels
		self.hidden_channels = hidden_channels

		# Network for retrieving scale and shift parameters as described in the paper
		self.net = nn.Sequential(
			weight_norm(nn.Conv2d(in_channels // 2, hidden_channels, kernel_size=3, padding='same')),
			nn.ReLU(),
			weight_norm(nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1)),
			nn.ReLU(),
			Conv2dZeroInit(hidden_channels, in_channels, kernel_size=3, padding='same')
			# Zero init to have scale=1 and shift=0. Cannot weight norm however due to decomposition resulting in division by zero.
		)
		self.rescale = weight_norm(Rescale(in_channels // 2))

	def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		# Because of the invertible convolutions that shuffle the channels, we can always split in the same way
		z_id, z_update = torch.chunk(x, 2, dim=1)

		params = self.net(z_id)
		log_s, t = torch.chunk(params, 2, dim=1)  # chunk along channel dimension
		log_s = self.rescale(torch.tanh(log_s))
		z_update = z_update * torch.exp(log_s) + t

		logdet = log_s.sum(dim=[1, 2, 3])

		return torch.cat([z_id, z_update], 1), logdet

	@torch.no_grad()
	def inverse(self, z: torch.Tensor) -> torch.Tensor:
		z_id, z_update = torch.chunk(z, 2, dim=1)

		log_s, t = self.net(z_id).chunk(2, 1)
		log_s = self.rescale(torch.tanh(log_s))
		z_update = (z_update - t) * torch.exp(-log_s)

		return torch.cat([z_id, z_update], dim=1)
