import tensorflow as tf

from vision_transformer.model.custom_layers import AddPositionalEmbeddings


def build_image_patches_encoder(patch_size, hidden_size, inputs):
    """The conv layer returns a tensor of shape (bs, n_patches_height, n_patches_width, n_filters).
    This is reshaped to (bs, n_patches_height * n_patches_width, n_filters). It is basically
    flattened along the spatial dimension.
    """
    encoded_patches = tf.keras.layers.Conv2D(filters=hidden_size, kernel_size=patch_size, strides=patch_size, padding='valid')(inputs)
    row_axis, col_axis = 1, 2  # channels last
    seq_len = (inputs.shape[row_axis] // patch_size) * (inputs.shape[col_axis] // patch_size)
    encoded_patches = tf.keras.layers.Reshape(target_shape=[seq_len, hidden_size])(encoded_patches)
    return encoded_patches


def build_mlp_block(inputs, mlp_dim, dropout_rate=0.1):
    """ Dense -> Dropout -> Dense -> Dropout
    The default dropout rate is from the original paper. Just like
    the gelu activation is.
    """
    x = tf.keras.layers.Dense(units=mlp_dim, activation=tf.nn.gelu)(inputs)
    x = tf.keras.layers.Dropout(rate=dropout_rate)(x)
    x = tf.keras.layers.Dense(units=inputs.shape[-1], activation=tf.nn.gelu)(x)
    x = tf.keras.layers.Dropout(rate=dropout_rate)(x)
    return x


def vit_transformer_block(num_heads, mlp_dim, inputs):
    """ layer_norm -> MultiHeadAttention -> residual_1 -> layer_norm -> mlp_block -> residual_2 """
    x0 = tf.keras.layers.LayerNormalization(dtype=inputs.dtype)(inputs)
    # MultiHeadAttention expects 2 arguments: the query and value. Optionally you could add a third argument
    # that is the key, but if not given, than key is the same as value.
    x0 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=inputs.shape[-1], dropout=0.1)(x0, x0)

    x0 = tf.keras.layers.Add()([x0, inputs])  # 1st residual part

    x1 = tf.keras.layers.LayerNormalization(dtype=x0.dtype)(x0)
    x1 = build_mlp_block(x1, mlp_dim)

    x_2 = tf.keras.layers.Add()([x1, x0])  # 2nd residual part
    return x_2


def vit_encoder(num_layers, mlp_dim, num_heads, inputs):
    """Combines the positional embeddings with the transformer blocks.
    """
    x = AddPositionalEmbeddings(name='add_positional_embeddings')(inputs)
    x = tf.keras.layers.Dropout(rate=0.2)(x)
    for _ in range(num_layers):
        x = vit_transformer_block(num_heads, mlp_dim, x)

    encoded = tf.keras.layers.LayerNormalization(name='encoder_norm')(x)
    return encoded
