import tensorflow as tf
from image_captioning.model.attention_layers import BahdanauAttention


class RNN_Decoder(tf.keras.Model):
    """Applies attention with a GRU layer to determine what the
    next word in the text should be.

    Args:
        embedding_dim (int):
        units (int):
        vocab_size (int):
    """

    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

        self.attention = BahdanauAttention(self.units)

    def call(self, x, features, hidden):
        """Note that embedding_dim = hidden_size.

        Args:
            x (tf.Tensor): words, shape (bs, 1). Each entry is a word represented by an integer.
            features (tf.Tensor): output of encoder shape (bs, H*W, embedding_dim)
            hidden (tf.Tensor): shape (bs, units)

        Returns:
            tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
                with shapes (bs, vocab_size), (bs, units), (bs, H*W, 1) and meanings
                probability for each word,
                state to pass to next step in the sequence,
                attention weights that can be used for plotting
        """
        # defining attention as a separate model
        context_vector, attention_weights = self.attention(features, hidden)

        # x shape after passing through embedding == (bs, 1, embedding_dim)
        x = self.embedding(x)  # word index -> embedding

        # x shape after concatenation == (bs, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        # output == (bs, 1, embedding_dim + hidden_size)
        # state == (bs, embedding_dim + hidden_size)
        output, state = self.gru(x)

        # shape == (bs, max_length, hidden_size); if multiple words are passed in one step
        # shape == (bs, 1, units)
        x = self.fc1(output)

        # x shape == (bs * max_length, hidden_size)
        # x shape == (bs, units); squeeze out second dimension
        x = tf.reshape(x, (-1, x.shape[2]))

        # output shape == (bs * max_length, vocab_size)
        # output shape == (bs, vocab_size); probability for each word in the vocabulary
        x = self.fc2(x)

        return x, state, attention_weights

    def reset_state(self, batch_size):
        """Used to initialize the hidden state for a each sample in a batch.

        Args:
            batch_size (int):

        Returns:
            tf.Tensor
        """
        return tf.zeros((batch_size, self.units))
