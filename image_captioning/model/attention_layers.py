import tensorflow as tf


class BahdanauAttention(tf.keras.Model):
    """
    Bahdanau Attention Mechanism:
    https://machinelearningmastery.com/the-bahdanau-attention-mechanism/

    Args:
        units (int): size of hidden dimension
    """

    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        """

        Args:
            features (tf.Tensor): shape (bs, H*W, embedding_dim)
            hidden (tf.Tensor): shape (bs, hidden_size)

        Returns:
            tuple[tf.Tensor, tf.Tensor]: context vector shape (bs, hidden_size), attention weights shape (bs, H*W, 1)
        """
        # features shape (bs, H*W, embedding_dim)
        # hidden shape (bs, hidden_size)

        # hidden_with_time_axis shape == (bs, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # self.W1(features) shape (bs, H*W, units)
        # self.W2(hidden_with_time_axis) shape (bs, 1, units); implicitly broadcasted to match H*W
        # attention_hidden_layer shape == (bs, H*W, units)
        attention_hidden_layer = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))

        # score shape == (bs, H*W, 1)
        # This gives you an unnormalized score for each image patch.
        score = self.V(attention_hidden_layer)

        # attention_weights shape == (bs, H*W, 1)
        # Every feature vector gets a score. Since a feature vector represents an image patch, each image patch gets an importance score/weight
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (bs, hidden_size)
        context_vector = attention_weights * features  # shape (bs, H*W, hidden_size); attention_weights are broadcasted to match features last dimension
        context_vector = tf.reduce_sum(context_vector, axis=1)  # reduce along spatial dimension

        return context_vector, attention_weights
