from typing import Any

import torch
import torch.nn as nn
from torch.distributions import Normal


class Net(nn.Module):
	"""Simple MLP for generating scale and shift parameters.
	"""

	def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, n_hidden_layers: int = 1, act: Any = nn.GELU):
		super().__init__()
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.output_dim = output_dim
		self.n_hidden_layers = n_hidden_layers
		self.act = act
		self.net = self.build_network()

	def build_network(self):
		layers = [
			nn.Linear(self.input_dim, self.hidden_dim),
			self.act()
		]
		for i in range(self.n_hidden_layers):
			layers += [
				nn.Linear(self.hidden_dim, self.hidden_dim),
				self.act()
			]
		layers.append(nn.Linear(self.hidden_dim, self.output_dim))
		net = nn.Sequential(*layers)
		return net

	def forward(self, x: torch.Tensor):
		return self.net(x)


class CouplingBijection1D(nn.Module):

	def __init__(self, input_dim, hidden_dim=128, n_hidden_layers=1):
		super().__init__()
		assert input_dim % 2 == 0, "Must provide an even number of input features."
		self.input_dim = input_dim
		self.input_dim_net = input_dim // 2  # half of the features will be used for finding scale and shift parameters
		self.output_dim_net = self.input_dim_net * 2  # each feature will get 2 parameters: a scale and a shift

		# Neural network for computing scaling and translation parameters
		self.net = Net(self.input_dim_net, hidden_dim, self.output_dim_net, n_hidden_layers)
        
	def forward(self, x: torch.Tensor):
		x0, x1 = torch.chunk(x, 2, dim=-1)
		z0 = x0  # for completeness, we add this mapping
		p = self.net(x0)
		log_s, b = torch.chunk(p, 2, dim=-1)
		z1 = x1 * log_s.exp() + b
		z = torch.cat([z0, z1], dim=-1)
		ldj = log_s.sum(-1)
		return z, ldj

	@torch.no_grad()
	def inverse(self, z: torch.Tensor):
		z0, z1 = torch.chunk(z, 2, dim=-1)
		x0 = z0  # for completeness, we add this mapping
		p = self.net(z0)
		log_s, b = torch.chunk(p, 2, dim=-1)
		x1 = (z1 - b) * (-log_s).exp()
		x = torch.cat([x0, x1], dim=-1)
		return x


class ReverseBijection1D(nn.Module):

	def __init__(self):
		super().__init__()

	def forward(self, x: torch.Tensor):
		return x.flip(dims=[-1]), x.new_zeros(x.shape[0])
    
	def inverse(self, z: torch.Tensor):
		return z.flip(dims=[-1])


class CouplingFlow1D(nn.Module):

	def __init__(self, input_dim: int, hidden_dim: int = 128, n_hidden_layers: int = 1, n_coupling_layers: int = 4):
		super().__init__()
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.n_hidden_layers = n_hidden_layers
		self.n_coupling_layers = n_coupling_layers

		self.loc = nn.Parameter(torch.zeros(input_dim), requires_grad=False)
		self.scale = nn.Parameter(torch.ones(input_dim), requires_grad=False)
		self.layers = self.build_model()

	def build_model(self):
		layers = []
		for i in range(self.n_coupling_layers - 1):
			layers.append(CouplingBijection1D(self.input_dim, self.hidden_dim, self.n_hidden_layers))
			layers.append(ReverseBijection1D())
		layers.append(CouplingBijection1D(self.input_dim, self.hidden_dim, self.n_hidden_layers))
		return nn.ModuleList(layers)

	@property
	def base_dist(self):
		return Normal(
			loc=self.loc,
			scale=self.scale,
		)
        
	def forward(self, x: torch.Tensor):
		"""
		Forward pass through all coupling layers.

		Args:
		    x (torch.Tensor): Input tensor of shape (batch_size, input_dim)

		Returns:
		    tuple: (transformed_x, total_log_det_jacobian)
		"""
		z = x
		total_log_det = torch.zeros(x.shape[0], device=x.device)

		for layer in self.layers:
			z, log_det = layer(z)
			total_log_det = total_log_det + log_det

		return z, total_log_det
    
	def inverse(self, z: torch.Tensor):
		"""
		Inverse pass through all coupling layers.

		Args:
		    z (torch.Tensor): Input tensor of shape (batch_size, input_dim)

		Returns:
		    tuple: (inverse_x, total_log_det_jacobian)
		"""
		x = z

		for layer in reversed(self.layers):
			x = layer.inverse(x)

		return x

	def log_prob(self, x: torch.Tensor):
		z, total_log_det = self.forward(x)
		log_prob = self.base_dist.log_prob(z).sum(1) + total_log_det
		return log_prob

	@torch.no_grad()
	def sample(self, n_samples: int):
		z = self.base_dist.sample((n_samples,))
		x = self.inverse(z)
		return x
