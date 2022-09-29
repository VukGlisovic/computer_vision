import tensorflow as tf


class BlazeBlock(tf.keras.layers.Layer):
    """
    BlazeBlock as described in
    "BlazeFace: Sub-millisecond Neural Face Detection on Mobile GPUs" (https://arxiv.org/pdf/1907.05047.pdf)

    Meaning a higher receptive field is used by applying a kernel size of 5x5.

    Args:
        stride (int): whether to downsample or not.
        filters (int): number of output feature maps.
    """

    def __init__(self, stride, filters):
        super(BlazeBlock, self).__init__()
        assert stride == 1 or stride == 2, "stride can only be 1 or 2. Other values are not allowed."
        self.stride = stride
        self.filters = filters
        # left path
        self.depthwise_conv = tf.keras.layers.DepthwiseConv2D(kernel_size=(5, 5), strides=self.stride, padding="same")
        self.conv2d_left = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=(1, 1), padding="same")
        # right path
        self.max_pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2))
        self.conv2d_right = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=(1, 1), padding="same")
        # combine left and right
        self.add = tf.keras.layers.Add()
        self.output_act = tf.keras.layers.Activation('relu')

    def call(self, inputs, *args, **kwargs):
        y = inputs
        x = self.depthwise_conv(inputs)
        x = self.conv2d_left(x)
        if self.stride == 2:
            y = self.max_pool(y)
            y = self.conv2d_right(y)
        output = self.add([x, y])
        return self.output_act(output)


class DoubleBlazeBlock(tf.keras.layers.Layer):
    """
    DoubleBlazeBlock as described in
    "BlazeFace: Sub-millisecond Neural Face Detection on Mobile GPUs" (https://arxiv.org/pdf/1907.05047.pdf)

    Meaning a higher receptive field is used by applying a kernel size of 5x5
    and in addition an extra conv layer is added in the left path.

    Args:
        stride (int): whether to downsample or not.
        filters1 (int): number of intermediate feature maps.
        filters2 (int): number of output feature maps.
    """

    def __init__(self, stride, filters1, filters2):
        super(DoubleBlazeBlock, self).__init__()
        assert stride == 1 or stride == 2, "stride can only be 1 or 2. Other values are not allowed."
        self.stride = stride
        self.filters1 = filters1
        self.filters2 = filters2
        # left path
        self.depthwise_conv1 = tf.keras.layers.DepthwiseConv2D(kernel_size=(5, 5), strides=self.stride, padding="same")
        self.conv2d_left1 = tf.keras.layers.Conv2D(filters=self.filters1, kernel_size=(1, 1), padding="same")
        self.act = tf.keras.layers.Activation('relu')
        self.depthwise_conv2 = tf.keras.layers.DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding="same")
        self.conv2d_left2 = tf.keras.layers.Conv2D(filters=self.filters2, kernel_size=(1, 1), padding="same")
        # right path
        self.max_pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2))
        self.conv2d_right = tf.keras.layers.Conv2D(filters=self.filters2, kernel_size=(1, 1), padding="same")
        # combine left and right
        self.add = tf.keras.layers.Add()
        self.output_act = tf.keras.layers.Activation('relu')

    def call(self, inputs, *args, **kwargs):
        y = inputs
        x = self.depthwise_conv1(inputs)
        x = self.conv2d_left1(x)
        x = self.act(x)
        x = self.depthwise_conv2(x)
        x = self.conv2d_left2(x)
        if self.stride == 2:
            y = self.max_pool(y)
            y = self.conv2d_right(y)
        output = self.add([x, y])
        return self.output_act(output)
