import torch
import torch.nn as nn
from torch.distributions import Normal


class Net(nn.Module):
	"""Simple MLP for generating scale and shift parameters.
	"""

	def __init__(self, input_dim, hidden_dim, output_dim, n_hidden_layers=1, act=nn.GELU):
		super().__init__()
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.output_dim = output_dim
		self.n_hidden_layers = n_hidden_layers
		self.act = act
		self.net = self.build_network()

	def build_network(self):
		net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            self.act(),
            *[nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                self.act()
            ) for _ in range(self.n_hidden_layers)],
            nn.Linear(self.hidden_dim, self.output_dim)
        )
		return net

	def forward(self, x):
		return self.net(x)


class CouplingBijection1D(nn.Module):

    def __init__(self, input_dim, hidden_dim=64, n_hidden_layers=3):
        super().__init__()
        assert input_dim % 2 == 0, "Must provide an even number of input features."
        self.input_dim = input_dim
        self.input_dim_net = input_dim // 2
        self.output_dim_net = input_dim

        # Neural network for computing scaling and translation parameters
        self.net = Net(self.input_dim_net, hidden_dim, self.output_dim_net, n_hidden_layers)
        
    def forward(self, x):
        x0, x1 = torch.chunk(x, 2, dim=-1)
        z0 = x0  # for completeness, we add this mapping
        p = self.net(x0)
        log_s, b = torch.chunk(p, 2, dim=-1)
        z1 = x1 * log_s.exp() + b
        z = torch.cat([z0, z1], dim=-1)
        ldj = log_s.sum(-1)
        return z, ldj

    @torch.no_grad()
    def inverse(self, z):
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

    def forward(self, x):
        return x.flip(dims=[-1]), x.new_zeros(x.shape[0])
    
    def inverse(self, z):
        return z.flip(dims=[-1])


class CouplingFlow1D(nn.Module):

    def __init__(self, input_dim, hidden_dim=64, n_hidden_layers=1, n_coupling_layers=4, device='cpu'):
        """
        Initialize a multi-layer coupling flow model.
        
        Args:
            input_dim (int): Dimension of the input tensor
            num_coupling_layers (int): Number of coupling layers to stack
            hidden_dim (int): Hidden dimension of the neural network
            num_layers_per_coupling (int): Number of layers in each coupling layer's network
        """
        super().__init__()
        self.input_dim = input_dim
        self.device = device
        
        # Create alternating coupling layers
        layers = []
        for i in range(n_coupling_layers - 1):
            layers.append(CouplingBijection1D(input_dim, hidden_dim, n_hidden_layers))
            layers.append(ReverseBijection1D())
        layers.append(CouplingBijection1D(input_dim, hidden_dim, n_hidden_layers))
        self.layers = nn.ModuleList(layers)

    @property
    def base_dist(self):
        return Normal(
            loc=torch.zeros(self.input_dim, device=self.device),
            scale=torch.ones(self.input_dim, device=self.device),
        )
        
    def forward(self, x):
        """
        Forward pass through all coupling layers.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
            
        Returns:
            tuple: (transformed_x, total_log_det_jacobian)
        """
        z = x
        total_log_det = 0
        
        for layer in self.layers:
            z, log_det = layer(z)
            total_log_det = total_log_det + log_det
            
        return z, total_log_det
    
    def inverse(self, z):
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

    def log_prob(self, x):
        z, total_log_det = self.forward(x)
        log_prob = self.base_dist.log_prob(z).sum(1) + total_log_det
        return log_prob

    @torch.no_grad()
    def sample(self, n_samples):
        z = self.base_dist.sample((n_samples,))
        for layer in reversed(self.layers):
            z = layer.inverse(z)
        x = z  # for completeness, we'll add this mapping
        return x
