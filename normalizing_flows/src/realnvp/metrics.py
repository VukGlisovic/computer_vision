import torch
import math
from typing import Union, Tuple


def bits_per_dim(nll: torch.Tensor, input_shape: Union[torch.Size, Tuple[int, ...]]) -> float:
    """
    Calculate bits per dimension metric for a batch of samples. Note the addition
    of log2(256) which is a consequence of the dequantization of the input pixels.
    
    Args:
        nll: Tensor of shape (batch_size,) containing the average negative log-likelihoods.
        input_shape: Shape of the input tensor (batch_size, channels, height, width).
    
    Returns:
        float: Average bits per dimension across the batch
    """
    # Calculate total number of dimensions (excluding batch dimension)
    num_dims = math.prod(input_shape[1:])
    
    # Convert log_prob to bits (divide by log(2))
    bits = nll / math.log(2)
    
    # Calculate bits per dimension
    bpd = bits / num_dims

    # Correct for dequantization step (256 bins in the image pixels)
    bpd = bpd.mean().item() + math.log(256, 2)

    return bpd
