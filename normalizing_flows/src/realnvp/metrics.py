import torch
import math
from typing import Union, Tuple


def bits_per_dim(log_prob: torch.Tensor, input_shape: Union[torch.Size, Tuple[int, ...]]) -> float:
    """
    Calculate bits per dimension metric for a batch of samples.
    
    Args:
        log_prob: Tensor of shape (batch_size,) containing log probabilities
        input_shape: Shape of the input tensor (batch_size, channels, height, width)
    
    Returns:
        float: Average bits per dimension across the batch
    """
    # Calculate total number of dimensions (excluding batch dimension)
    num_dims = math.prod(input_shape[1:])
    
    # Convert log_prob to bits (divide by log(2))
    bits = -log_prob / math.log(2)
    
    # Calculate bits per dimension
    bits_per_dim = bits / num_dims
    
    # Return average across batch
    return bits_per_dim.mean().item()
