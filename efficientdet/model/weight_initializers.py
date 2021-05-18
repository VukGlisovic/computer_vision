import numpy as np
from tensorflow.keras.initializers import Initializer


class PriorProbability(Initializer):
    """Mainly used to set a prior probability for the bias term
    in certain layers.

    Args:
        probability (float): prior probability
    """

    def __init__(self, probability=0.01):
        self.probability = probability

    def __call__(self, shape, dtype=None, **kwargs):
        """Sets the bias to -log((1 - p)/p) for the foreground class.

        Args:
            shape (tuple):
            dtype (str):
            **kwargs (dict):

        Returns:
            np.ndarray
        """
        result = np.ones(shape, dtype=np.float32) * -np.log((1 - self.probability) / self.probability)
        return result

    def get_config(self):
        """Sets the config method.

        Returns:
            dict
        """
        return {'probability': self.probability}
