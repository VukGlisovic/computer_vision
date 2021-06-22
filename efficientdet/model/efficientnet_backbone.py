import math
import string
import tensorflow as tf
from tensorflow.keras import layers


DEFAULT_BLOCKS = [
    dict(kernel_size=3, num_repeat=1, input_filters=32, output_filters=16, expand_ratio=1, id_skip=True, strides=[1, 1], se_ratio=0.25),
    dict(kernel_size=3, num_repeat=2, input_filters=16, output_filters=24, expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    dict(kernel_size=5, num_repeat=2, input_filters=24, output_filters=40, expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    dict(kernel_size=3, num_repeat=3, input_filters=40, output_filters=80, expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    dict(kernel_size=5, num_repeat=3, input_filters=80, output_filters=112, expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25),
    dict(kernel_size=5, num_repeat=4, input_filters=112, output_filters=192, expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    dict(kernel_size=3, num_repeat=1, input_filters=192, output_filters=320, expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25),
]


def round_filters(filters, width_coefficient, depth_divisor):
    """Round number of filters based on width multiplier.

    Args:
        filters (int):
        width_coefficient (float):
        depth_divisor (float):

    Returns:
        int
    """
    filters *= width_coefficient
    new_filters = int(filters + depth_divisor / 2) // depth_divisor * depth_divisor
    new_filters = max(depth_divisor, new_filters)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor
    return int(new_filters)


def round_repeats(repeats, depth_coefficient):
    """Round number of repeats based on depth multiplier.

    Args:
        repeats (int):
        depth_coefficient (float)

    Returns:
        int
    """
    return int(math.ceil(depth_coefficient * repeats))


def mb_conv_block(inputs,
                  kernel_size,
                  input_filters,
                  output_filters,
                  expand_ratio,
                  id_skip,
                  strides,
                  se_ratio,
                  activation,
                  drop_rate=None,
                  prefix='',
                  **kwargs):
    """Mobile Inverted Residual Bottleneck.

    Args:inputs,
        kernel_size (int):
        input_filters (int):
        output_filters (int):
        expand_ratio (int):
        id_skip (bool):
        strides (list[int]):
        se_ratio (float):
        activation (Union[callable, str]):
        drop_rate (float):
        prefix (str):
    """
    # Expansion phase
    filters = input_filters * expand_ratio
    if expand_ratio != 1:
        x = layers.Conv2D(filters=filters,
                          kernel_size=1,
                          padding='same',
                          use_bias=False,
                          name=prefix + 'expand_conv')(inputs)
        x = layers.BatchNormalization(name=prefix + 'expand_bn')(x)
        x = layers.Activation(activation, name=prefix + 'expand_activation')(x)
    else:
        x = inputs

    # Depthwise Convolution
    x = layers.DepthwiseConv2D(kernel_size=kernel_size,
                               strides=strides,
                               padding='same',
                               use_bias=False,
                               name=prefix + 'dwconv')(x)
    x = layers.BatchNormalization(name=prefix + 'bn')(x)
    x = layers.Activation(activation, name=prefix + 'activation')(x)

    # Squeeze and Excitation phase
    if (se_ratio is not None) and (0 < se_ratio <= 1):
        se_tensor = layers.GlobalAveragePooling2D(name=prefix + 'se_squeeze')(x)
        se_tensor = layers.Reshape((1, 1, filters), name=prefix + 'se_reshape')(se_tensor)

        num_reduced_filters = max(1, int(input_filters * se_ratio))
        se_tensor = layers.Conv2D(filters=num_reduced_filters,
                                  kernel_size=1,
                                  activation=activation,
                                  padding='same',
                                  use_bias=True,
                                  name=prefix + 'se_reduce')(se_tensor)
        se_tensor = layers.Conv2D(filters=filters,
                                  kernel_size=1,
                                  activation='sigmoid',
                                  padding='same',
                                  use_bias=True,
                                  name=prefix + 'se_expand')(se_tensor)
        x = layers.multiply([x, se_tensor], name=prefix + 'se_excite')

    # Output phase
    x = layers.Conv2D(filters=output_filters,
                      kernel_size=1,
                      padding='same',
                      use_bias=False,
                      name=prefix + 'project_conv')(x)
    x = layers.BatchNormalization(name=prefix + 'project_bn')(x)
    if id_skip and all(s == 1 for s in strides) and input_filters == output_filters:
        if drop_rate and (drop_rate > 0):
            x = layers.Dropout(drop_rate, noise_shape=(None, 1, 1, 1), name=prefix + 'drop')(x)
        x = layers.add([x, inputs], name=prefix + 'add')

    return x


def efficientnet(width_coefficient,
                 depth_coefficient,
                 dropout_rate=0.2,
                 drop_connect_rate=0.2,
                 depth_divisor=8,
                 block_configs=DEFAULT_BLOCKS,
                 model_name='efficientnet',
                 return_fpn_features=False,
                 include_top=True,
                 input_shape=None,
                 input_tensor=None,
                 pooling=None,
                 classes=10,
                 **kwargs):
    """Instantiates the EfficientNet architecture using given scaling coefficients.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    # Arguments
        width_coefficient: float, scaling coefficient for network width.
        depth_coefficient: float, scaling coefficient for network depth.
        dropout_rate: float, dropout rate before final classifier layer.
        drop_connect_rate: float, dropout rate at skip connections.
        depth_divisor: int.
        blocks_args: A list of BlockArgs to construct block modules.
        model_name: string, model name.
        return_fpn_features: whether to return the intermediate features
            in a list.
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor
            (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False.
            It should have exactly 3 inputs channels.
        pooling: optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    """
    img_input = input_tensor if input_tensor is not None else layers.Input(shape=input_shape)

    activation = tf.keras.activations.swish

    # Build stem
    x = img_input
    x = layers.Conv2D(filters=round_filters(32, width_coefficient, depth_divisor),
                      kernel_size=3,
                      strides=(2, 2),
                      padding='same',
                      use_bias=False,
                      name='stem_conv')(x)
    x = layers.BatchNormalization(name='stem_bn')(x)
    x = layers.Activation(activation, name='stem_activation')(x)

    # Build blocks
    num_blocks_total = sum(block_args['num_repeat'] for block_args in block_configs)
    block_num = 0
    features = []
    for idx, block_config in enumerate(block_configs):
        # Update block input and output filters based on depth multiplier.
        block_config['input_filters'] = round_filters(block_config['input_filters'], width_coefficient, depth_divisor)
        block_config['output_filters'] = round_filters(block_config['output_filters'], width_coefficient, depth_divisor)
        block_config['num_repeat'] = round_repeats(block_config['num_repeat'], depth_coefficient)

        # The first block needs to take care of stride and filter size increase.
        drop_rate = drop_connect_rate * float(block_num) / num_blocks_total
        x = mb_conv_block(x,
                          activation=activation,
                          drop_rate=drop_rate,
                          prefix='block{}a_'.format(idx + 1),
                          **block_config)
        block_num += 1
        if block_config['num_repeat'] > 1:
            # make sure repeated blocks keep the same output shape
            block_config['input_filters'] = block_config['output_filters']
            block_config['strides'] = [1, 1]
            for bidx in range(block_config['num_repeat'] - 1):
                drop_rate = drop_connect_rate * float(block_num) / num_blocks_total
                block_prefix = 'block{}{}_'.format(idx + 1, string.ascii_lowercase[bidx + 1])
                x = mb_conv_block(x,
                                  activation=activation,
                                  drop_rate=drop_rate,
                                  prefix=block_prefix,
                                  **block_config)
                block_num += 1
        if idx < len(block_configs) - 1 and block_configs[idx + 1]['strides'][0] == 2:
            features.append(x)
        elif idx == len(block_configs) - 1:
            features.append(x)

    if return_fpn_features:
        return tf.keras.Model(inputs=img_input, outputs=features)

    # Build top
    x = layers.Conv2D(filters=round_filters(1280, width_coefficient, depth_divisor),
                      kernel_size=1,
                      padding='same',
                      use_bias=False,
                      name='top_conv')(x)
    x = layers.BatchNormalization(name='top_bn')(x)
    x = layers.Activation(activation, name='top_activation')(x)
    if include_top:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        if dropout_rate and dropout_rate > 0:
            x = layers.Dropout(dropout_rate, name='top_dropout')(x)
        x = layers.Dense(classes, activation='softmax', name='probs')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D(name='max_pool')(x)

    # Create model.
    model = tf.keras.models.Model(img_input, x, name=model_name)

    return model
