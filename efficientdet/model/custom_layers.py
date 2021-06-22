from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf


class ConvBlock(layers.Layer):
    """A simple block that chains a Conv2D and a BatchNormalization
    layer together.

    Args:
        num_channels (int):
        kernel_size (int):
        strides (int):
    """

    def __init__(self, num_channels, kernel_size, strides, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = layers.Conv2D(num_channels, kernel_size=kernel_size, strides=strides, padding='same', use_bias=True)
        self.bn = layers.BatchNormalization()

    def call(self, x, **kwargs):
        """Applies: Conv2D -> BatchNormalization

        Args:
            x (tf.Tensor):

        Returns:
            tf.Tensor:
        """
        x = self.bn(self.conv(x))
        return x


class SeparableConvBlock(layers.Layer):
    """A simple block that chains a SeparableConv2D and a BatchNormalization
    layer together.

    Args:
        num_channels (int):
        kernel_size (int):
        strides (int):
    """

    def __init__(self, num_channels, kernel_size, strides, **kwargs):
        super(SeparableConvBlock, self).__init__()
        self.sep_conv = layers.SeparableConv2D(num_channels, kernel_size=kernel_size, strides=strides, padding='same', use_bias=True)
        self.bn = layers.BatchNormalization()

    def call(self, x, **kwargs):
        """Applies: SeparableConv2D -> BatchNormalization

        Args:
            x (tf.Tensor):

        Returns:
            tf.Tensor:
        """
        x = self.bn(self.sep_conv(x))
        return x


class FastNormalizedFusion(keras.layers.Layer):

    def __init__(self, epsilon=1e-4, **kwargs):
        super(FastNormalizedFusion, self).__init__(**kwargs)
        self.epsilon = epsilon
        self.relu = layers.Activation('relu')

    def build(self, input_shape):
        self.num_in = len(input_shape)  # expected num_in is 2 or 3
        self.w = self.add_weight(name=self.name,
                                 shape=(self.num_in,),
                                 initializer=keras.initializers.ones(),
                                 trainable=True,
                                 dtype=tf.float32)

    def call(self, inputs, **kwargs):
        hxw = tf.shape(inputs[0])[1:3]
        input_resized = tf.image.resize(inputs[-1], size=hxw, method='nearest')  # up or downsamples
        inputs = inputs[:-1] + [input_resized]
        w = self.relu(self.w)  # make sure weights are positive
        x = tf.reduce_sum([w[i] * inputs[i] for i in range(self.num_in)], axis=0)
        w_sum = tf.reduce_sum(w)
        x = x / (w_sum + self.epsilon)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super(FastNormalizedFusion, self).get_config()
        config.update(
            {'epsilon': self.epsilon}
        )
        return config


class BiFPNFeatureFusion(layers.Layer):

    def __init__(self, num_channels, **kwargs):
        super(BiFPNFeatureFusion, self).__init__(**kwargs)
        self.num_channels = num_channels
        self.fast_normalized_fusion = FastNormalizedFusion(**kwargs)
        self.activation = layers.Activation(tf.nn.swish)
        self.sep_conv_block = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1)

    def call(self, inputs, **kwargs):
        features = self.fast_normalized_fusion(inputs)
        features = self.activation(features)
        features = self.sep_conv_block(features)
        return features

    def get_config(self):
        config = super(BiFPNFeatureFusion, self).get_config()
        config.update(
            {'num_channels': self.num_channels}
        )
        return config


class BiFPNBlock(layers.Layer):

    def __init__(self, num_channels, add_conv_blocks, id, **kwargs):
        super(BiFPNBlock, self).__init__(**kwargs)
        self.num_channels = num_channels
        self.add_conv_blocks = add_conv_blocks
        self.id = id
        # downward path
        self.bifpnff_6_td = BiFPNFeatureFusion(num_channels, name=f'fpn_cells/cell_{id}/fnode0/add')
        self.bifpnff_5_td = BiFPNFeatureFusion(num_channels, name=f'fpn_cells/cell_{id}/fnode1/add')
        self.bifpnff_4_td = BiFPNFeatureFusion(num_channels, name=f'fpn_cells/cell_{id}/fnode2/add')
        if self.add_conv_blocks:
            self.conv_block_p5_in_1 = ConvBlock(num_channels, kernel_size=1, strides=1)
            self.conv_block_p4_in_1 = ConvBlock(num_channels, kernel_size=1, strides=1)
        # upward path
        self.bifpnff_3_out = BiFPNFeatureFusion(num_channels, name=f'fpn_cells/cell_{id}/fnode3/add')
        self.bifpnff_4_out = BiFPNFeatureFusion(num_channels, name=f'fpn_cells/cell_{id}/fnode4/add')
        self.bifpnff_5_out = BiFPNFeatureFusion(num_channels, name=f'fpn_cells/cell_{id}/fnode5/add')
        self.bifpnff_6_out = BiFPNFeatureFusion(num_channels, name=f'fpn_cells/cell_{id}/fnode6/add')
        self.bifpnff_7_out = BiFPNFeatureFusion(num_channels, name=f'fpn_cells/cell_{id}/fnode7/add')
        if self.add_conv_blocks:
            self.conv_block_p3_in = ConvBlock(num_channels, kernel_size=1, strides=1)
            self.conv_block_p4_in_2 = ConvBlock(num_channels, kernel_size=1, strides=1)
            self.conv_block_p5_in_2 = ConvBlock(num_channels, kernel_size=1, strides=1)

    def call(self, inputs, **kwargs):
        P3_in, P4_in, P5_in, P6_in, P7_in = inputs
        if self.add_conv_blocks:
            P3_in = self.conv_block_p3_in(P3_in)
            P4_in_1 = self.conv_block_p4_in_1(P4_in)
            P4_in_2 = self.conv_block_p4_in_2(P4_in)
            P5_in_1 = self.conv_block_p5_in_1(P5_in)
            P5_in_2 = self.conv_block_p5_in_2(P5_in)
        else:
            P4_in_1 = P4_in
            P4_in_2 = P4_in
            P5_in_1 = P5_in
            P5_in_2 = P5_in
        # downward path
        P6_td = self.bifpnff_6_td([P6_in, P7_in])
        P5_td = self.bifpnff_5_td([P5_in_1, P6_td])
        P4_td = self.bifpnff_4_td([P4_in_1, P5_td])
        # upward path
        P3_out = self.bifpnff_3_out([P3_in, P4_td])
        P4_out = self.bifpnff_4_out([P4_in_2, P4_td, P3_out])
        P5_out = self.bifpnff_5_out([P5_in_2, P5_td, P4_out])
        P6_out = self.bifpnff_6_out([P6_in, P6_td, P5_out])
        P7_out = self.bifpnff_7_out([P7_in, P6_out])
        return P3_out, P4_out, P5_out, P6_out, P7_out

    def get_config(self):
        config = super(BiFPNBlock, self).get_config()
        config.update(
            {'id': self.id, 'num_channels': self.num_channels}
        )
        return config


class BiFeaturePyramid(layers.Layer):

    def __init__(self, n_blocks, num_channels, **kwargs):
        super(BiFeaturePyramid, self).__init__(**kwargs)
        self.n_blocks = n_blocks
        self.num_channels = num_channels
        self.create_p6_in = ConvBlock(num_channels, kernel_size=1, strides=1)
        self.max_pool_p6_in = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')
        self.max_pool_p7_in = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')
        self.fpn_blocks = [BiFPNBlock(num_channels, add_conv_blocks=True, id=0)]
        self.fpn_blocks += [BiFPNBlock(num_channels, add_conv_blocks=False, id=i) for i in range(1, n_blocks + 1)]

    def call(self, inputs, **kwargs):
        _, _, P3, P4, P5 = inputs
        P6 = self.max_pool_p6_in(self.create_p6_in(P5))
        P7 = self.max_pool_p7_in(P6)
        features = [P3, P4, P5, P6, P7]
        for block in self.fpn_blocks:
            features = block(features)
        return features

    def get_config(self):
        config = super(BiFeaturePyramid, self).get_config()
        config.update(
            {'n_blocks': self.n_blocks, 'num_channels': self.num_channels}
        )
        return config
