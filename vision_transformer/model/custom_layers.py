import tensorflow as tf


class AddPositionalEmbeddings(tf.keras.layers.Layer):
    """inputs are image patches
    Custom layer to add positional embeddings to the inputs.
    """

    def __init__(self, pos_embedding_init=tf.keras.initializers.RandomNormal(stddev=0.02), **kwargs):
        super().__init__(**kwargs)
        self.pos_embedding_init = pos_embedding_init
        self.pos_embedding = None  # is created in the build

    def build(self, inputs_shape):
        """input_shape = (batch_size, seq_len, emb_dim). We want each
        sample in the batch to get the same positional embeddings.
        """
        pos_emb_shape = (1, inputs_shape[1], inputs_shape[2])
        self.pos_embedding = self.add_weight(name='positional_embedding',
                                             shape=pos_emb_shape,
                                             initializer=self.pos_embedding_init)

    def call(self, inputs):
        """inputs.shape = (batch_size, seq_len, emb_dim). Thus, self.pos_embedding
        is broadcasted to match the batch size.
        """
        pos_embedding = tf.cast(self.pos_embedding, inputs.dtype)
        return inputs + pos_embedding

    def get_config(self):
        """Needed to be able to save and load the model again.
        """
        config = super().get_config()
        config.update({
            'pos_embedding_init': self.pos_embedding_init
        })
        return config
