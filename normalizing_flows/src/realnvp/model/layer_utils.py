from typing import Any

import torch.nn as nn


def weight_norm(layer: Any) -> Any:
    """The reason this method exists, is for documentation. The idea is taken from:

    "Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks" - https://arxiv.org/abs/1602.07868

    The idea is that it stabilizes training by decomposing the weights into a
    direction vector and a magnitude scalar. These individual components are then
    updated through backpropagation.
    """
    return nn.utils.parametrizations.weight_norm(layer)
