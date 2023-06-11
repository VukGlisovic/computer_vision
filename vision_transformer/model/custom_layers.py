import tensorflow as tf


def generate_patch_conv_orgPaper_f(patch_size, hidden_size, inputs):
    """The conv layer returns a tensor of shape (bs, n_patches_height, n_patches_width, n_filters).
    This is reshaped to (bs, n_patches_height * n_patches_width, n_filters). It is basically
    flattened along the spatial dimension.
    """
    encoded_patches = tf.keras.layers.Conv2D(filters=hidden_size, kernel_size=patch_size, strides=patch_size, padding='valid')(inputs)
    row_axis, col_axis = 1, 2  # channels last
    seq_len = (inputs.shape[row_axis] // patch_size) * (inputs.shape[col_axis] // patch_size)
    encoded_patches = tf.keras.layers.Reshape(target_shape=[seq_len, hidden_size])(encoded_patches)
    return encoded_patches


class AddPositionEmbs(tf.keras.layers.Layer):
    """inputs are image patches
    Custom layer to add positional embeddings to the inputs.
    """

    def __init__(self, pos_embedding_init=tf.keras.initializers.RandomNormal(stddev=0.02), **kwargs):
        super().__init__(**kwargs)
        self.pos_embedding_init = pos_embedding_init

    def build(self, inputs_shape):
        """input_shape = (batch_size, seq_len, emb_dim). We want each
        sample in the batch to get the same positional embeddings.
        """
        pos_emb_shape = (1, inputs_shape[1], inputs_shape[2])
        self.pos_embedding = self.add_weight(name='pos_embedding',
                                             shape=pos_emb_shape,
                                             initializer=self.pos_embedding_init)

    def call(self, inputs, inputs_positions=None):
        """inputs.shape = (batch_size, seq_len, emb_dim). Thus, self.pos_embedding
        is broadcasted to match the batch size.
        """
        pos_embedding = tf.cast(self.pos_embedding, inputs.dtype)
        return inputs + pos_embedding

    def get_config(self):
        config = super().get_config()
        config.update({
            'pos_embedding_init': self.pos_embedding_init
        })
        return config


def mlp_block_f(mlp_dim, inputs):
    x = tf.keras.layers.Dense(units=mlp_dim, activation=tf.nn.gelu)(inputs)
    x = tf.keras.layers.Dropout(rate=0.1)(x)  # dropout rate is from original paper,
    x = tf.keras.layers.Dense(units=inputs.shape[-1], activation=tf.nn.gelu)(x)  # check GELU paper
    x = tf.keras.layers.Dropout(rate=0.1)(x)
    return x


def Encoder1Dblock_f(num_heads, mlp_dim, inputs):
    x = tf.keras.layers.LayerNormalization(dtype=inputs.dtype)(inputs)
    x = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=inputs.shape[-1], dropout=0.1)(x, x)
    # self attention multi-head, dropout_rate is from original implementation
    x = tf.keras.layers.Add()([x, inputs])  # 1st residual part

    y = tf.keras.layers.LayerNormalization(dtype=x.dtype)(x)
    y = mlp_block_f(mlp_dim, y)
    y_1 = tf.keras.layers.Add()([y, x])  # 2nd residual part
    return y_1


def Encoder_f(num_layers, mlp_dim, num_heads, inputs):
    x = AddPositionEmbs(name='pos_embedding_input')(inputs)
    x = tf.keras.layers.Dropout(rate=0.2)(x)
    for _ in range(num_layers):
        x = Encoder1Dblock_f(num_heads, mlp_dim, x)

    encoded = tf.keras.layers.LayerNormalization(name='encoder_norm')(x)
    return encoded
