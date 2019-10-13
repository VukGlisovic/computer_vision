from tensorflow import keras


def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    """Creates a block with:
    Conv2D -> (BatchNormalization) -> Activation -> Conv2D -> (BatchNormalization) -> Activation

    Args:
        input_tensor (tf.Tensor):
        n_filters (int):
        kernel_size (int):
        batchnorm (bool):

    Returns:

    """
    # first layer
    x = keras.layers.Conv2D(filters=n_filters,
                            kernel_size=(kernel_size, kernel_size),
                            kernel_initializer="he_normal",
                            padding="same")(input_tensor)
    if batchnorm:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    # second layer
    x = keras.layers.Conv2D(filters=n_filters,
                            kernel_size=(kernel_size, kernel_size),
                            kernel_initializer="he_normal",
                            padding="same")(x)
    if batchnorm:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    return x


def create_unet_model(input_img, n_filters=16, dropout_rate=0.5, batchnorm=True):
    """This method generates the full architecture of the model. The
    comments show the down- and up-sampling changes if an input image
    with size (101, 101, ?) is given.

    Args:
        input_img (tf.Tensor):
        n_filters (int):
        dropout_rate (float):
        batchnorm (bool):

    Returns:
        keras.models.Model
    """
    # contracting path
    # 101 -> 50
    c1 = conv2d_block(input_img, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    p1 = keras.layers.MaxPool2D((2, 2), name='p1')(c1)
    p1 = keras.layers.Dropout(dropout_rate * 0.5, name='d1')(p1)

    # 50 -> 25
    c2 = conv2d_block(p1, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    p2 = keras.layers.MaxPool2D((2, 2), name='p2')(c2)
    p2 = keras.layers.Dropout(dropout_rate, name='d2')(p2)

    # 25 -> 12
    c3 = conv2d_block(p2, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    p3 = keras.layers.MaxPool2D((2, 2), name='p3')(c3)
    p3 = keras.layers.Dropout(dropout_rate, name='d3')(p3)

    # 12 -> 6
    c4 = conv2d_block(p3, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    p4 = keras.layers.MaxPool2D((2, 2), name='p4')(c4)
    p4 = keras.layers.Dropout(dropout_rate, name='d4')(p4)

    # middle
    c5 = conv2d_block(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)

    # expansive path
    # 6 -> 12
    u6 = keras.layers.Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
    u6 = keras.layers.concatenate([u6, c4])
    u6 = keras.layers.Dropout(dropout_rate, name='d6')(u6)
    c6 = conv2d_block(u6, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)

    # 12 -> 25
    # because of valid padding we get an output shape of (25, 25, ?)
    # input_shape * stride + (kernel_size - stride) = 12 * 2 + (3 - 2) = 25
    u7 = keras.layers.Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='valid')(c6)
    u7 = keras.layers.concatenate([u7, c3])
    u7 = keras.layers.Dropout(dropout_rate, name='d7')(u7)
    c7 = conv2d_block(u7, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)

    # 25 -> 50
    u8 = keras.layers.Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = keras.layers.concatenate([u8, c2])
    u8 = keras.layers.Dropout(dropout_rate, name='d8')(u8)
    c8 = conv2d_block(u8, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)

    # 50 -> 101
    u9 = keras.layers.Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='valid')(c8)
    u9 = keras.layers.concatenate([u9, c1], axis=3)
    u9 = keras.layers.Dropout(dropout_rate, name='d9')(u9)
    c9 = conv2d_block(u9, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)

    outputs = keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = keras.models.Model(inputs=[input_img], outputs=[outputs])
    return model
