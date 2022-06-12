import tensorflow as tf


def create_inception_v3():
    """Creates an InceptionV3 network that is initialized with
    imagenet weights.

    Returns:
        tf.keras.models.Model
    """
    return tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')


class CNN_Encoder(tf.keras.Model):
    """The CNN_Encoder expects the output features of a pretrained
    network (e.g. InceptionV3) where the spatial dimensions are
    flattened.

    This layer/model basically reduces the dimensionality of the
    input features.

    Args:
        embedding_dim (int):
    """
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        """
        Input shape: (bs, H*W, latent_dim)
        Output shape: (bs, H*W, embedding_dim)

        Args:
            x (tf.Tensor):

        Returns:
            tf.Tensor
        """
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x
