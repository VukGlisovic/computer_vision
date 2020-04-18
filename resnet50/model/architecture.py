import tensorflow as tf
from tensorflow import keras


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block has as input input_tensor and does the following:
    x, x_shortcut = input_tensor, input_tensor
    x = x -> Conv2D -> BatchNorm -> ReLU -> Conv2D -> BatchNorm -> ReLU -> Conv2D -> BatchNorm
    x = (x + x_shortcut) -> ReLU
    The batch normalization axis is 3; this should correspond to the channels
    axis.

    Args:
        input_tensor (tf.Tensor):
        kernel_size (int):
        filters (list[int]): a list with 3 integers
        stage (str): the stage this identity block belongs to
        block (str): the block descriptor within the stage

    Returns:
        tf.Tensor
    """
    assert len(filters) == 3, "List with number of filters should contain 3 filter numbers."
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + stage + block + '_branch'
    bn_name_base = 'bn' + stage + block + '_branch'

    x = keras.layers.Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2c')(x)

    x = keras.layers.add([x, input_tensor])
    x = keras.layers.Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """Does similar things as the identity block, but in addition applies
    a Conv2D and a BatchNorm to the shortcut. The purpose is also to shrink
    the height and width by a factor of 2.

    Args:
        input_tensor (tf.Tensor):
        kernel_size (int):
        filters (list[int]):
        stage (str):
        block (str):
        strides (tuple):

    Returns:
        tf.Tensor
    """
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = keras.layers.Conv2D(filters1, (1, 1), strides=strides, name=conv_name_base + '2a')(input_tensor)
    x = keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2c')(x)

    shortcut = keras.layers.Conv2D(filters3, (1, 1), strides=strides, name=conv_name_base + '1')(input_tensor)
    shortcut = keras.layers.BatchNormalization(axis=3, name=bn_name_base + '1')(shortcut)

    x = keras.layers.add([x, shortcut])
    x = keras.layers.Activation('relu')(x)
    return x


def res_net_50(input_shape, n_classes):
    """Creates a ResNet50 model from scratch.

    Args:
        input_shape (tuple):
        n_classes (int):

    Returns:
        keras.models.Model
    """
    input_tensor = keras.layers.Input(shape=input_shape)

    x = keras.layers.ZeroPadding2D((3, 3))(input_tensor)
    x = keras.layers.Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = keras.layers.BatchNormalization(axis=3, name='bn_conv1')(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage='2', block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage='2', block='b')
    x = identity_block(x, 3, [64, 64, 256], stage='2', block='c')

    x = conv_block(x, 3, [128, 128, 512], stage='3', block='a')
    x = identity_block(x, 3, [128, 128, 512], stage='3', block='b')
    x = identity_block(x, 3, [128, 128, 512], stage='3', block='c')
    x = identity_block(x, 3, [128, 128, 512], stage='3', block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage='4', block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage='4', block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage='4', block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage='4', block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage='4', block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage='4', block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage='5', block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage='5', block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage='5', block='c')

    x = keras.layers.AveragePooling2D((2, 2), name='avg_pool')(x)  # original pool size is 7x7

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(n_classes, activation='softmax', name='output')(x)

    # Create model.
    model = keras.models.Model(input_tensor, x, name='resnet50')
    return model
