from unet.model.constants import *
import tensorflow as tf
from tensorflow import keras
from unet.model.losses import get_loss_function
from unet.model.metrics import get_metric_function


def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    """Creates a block with:
    Conv2D -> (BatchNormalization) -> Activation -> Conv2D -> (BatchNormalization) -> Activation

    Args:
        input_tensor (tf.Tensor):
        n_filters (int):
        kernel_size (int):
        batchnorm (bool):

    Returns:
        tf.Tensor
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


def create_unet_model(input_img, n_filters=16, batchnorm=True, apply_final_activation=True):
    """This method generates the full architecture of the model. The
    comments show the down- and up-sampling changes if an input image
    with size (101, 101, ?) is given.

    Args:
        input_img (tf.Tensor):
        n_filters (int):
        batchnorm (bool):
        apply_final_activation (bool):

    Returns:
        keras.models.Model
    """
    # contracting path
    # 101 -> 50
    conv1 = conv2d_block(input_img, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    pool1 = keras.layers.MaxPool2D((2, 2), name='p1')(conv1)

    # 50 -> 25
    conv2 = conv2d_block(pool1, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    pool2 = keras.layers.MaxPool2D((2, 2), name='p2')(conv2)

    # 25 -> 12
    conv3 = conv2d_block(pool2, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    pool3 = keras.layers.MaxPool2D((2, 2), name='p3')(conv3)

    # 12 -> 6
    conv4 = conv2d_block(pool3, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    pool4 = keras.layers.MaxPool2D((2, 2), name='p4')(conv4)

    # middle
    conv_middle = conv2d_block(pool4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)

    # expansive path
    # 6 -> 12
    deconv4 = keras.layers.Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(conv_middle)
    deconv4 = keras.layers.concatenate([deconv4, conv4])
    deconv4 = conv2d_block(deconv4, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)

    # 12 -> 25
    # because of valid padding we get an output shape of (25, 25, ?)
    # input_shape * stride + (kernel_size - stride) = 12 * 2 + (3 - 2) = 25
    deconv3 = keras.layers.Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='valid')(deconv4)
    deconv3 = keras.layers.concatenate([deconv3, conv3])
    deconv3 = conv2d_block(deconv3, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)

    # 25 -> 50
    deconv2 = keras.layers.Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(deconv3)
    deconv2 = keras.layers.concatenate([deconv2, conv2])
    deconv2 = conv2d_block(deconv2, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)

    # 50 -> 101
    deconv1 = keras.layers.Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='valid')(deconv2)
    deconv1 = keras.layers.concatenate([deconv1, conv1], axis=3)
    deconv1 = conv2d_block(deconv1, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)

    outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation=None)(deconv1)
    if apply_final_activation:
        outputs = keras.layers.Activation('sigmoid')(outputs)
    model = keras.models.Model(inputs=[input_img], outputs=[outputs])
    return model


def get_unet_model(loss, metric, learning_rate=0.01, n_filters=16, batchnorm=True, apply_final_activation=True, compile=True, weights_path=None, **kwargs):
    """Adds the input tensor to the keras U-net model.

    Args:
        loss (str):
        metric (str):
        learning_rate (float):
        n_filters (int):
        batchnorm (bool):
        apply_final_activation (bool):
        weights_path (Union[str, None]):

    Returns:
        keras.models.Model
    """
    input_image = keras.layers.Input(IMAGE_SHAPE, name='image')
    model = create_unet_model(input_img=input_image,
                              n_filters=n_filters,
                              batchnorm=batchnorm,
                              apply_final_activation=apply_final_activation)

    if compile:
        optimizer = keras.optimizers.Adam(lr=learning_rate)
        loss_fnc = get_loss_function(loss)
        metric_fnc = get_metric_function(metric)
        model.compile(loss=loss_fnc, optimizer=optimizer, metrics=[metric_fnc])

    if weights_path is not None:
        if os.path.isfile(weights_path):
            model.load_weights(weights_path)
        elif os.path.isdir(weights_path):
            model.load_weights(tf.train.latest_checkpoint(weights_path))
        else:
            raise FileNotFoundError("Could not load model from path %s", weights_path)
    return model
