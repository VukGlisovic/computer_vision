import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import initializers
from efficientdet.constants import NUM_ANCHORS
from efficientdet.model.weight_initializers import PriorProbability


class BoxNet(models.Model):

    def __init__(self, width, depth, separable_conv=True, **kwargs):
        super(BoxNet, self).__init__(**kwargs)
        self.width = width
        self.depth = depth
        self.num_anchors = NUM_ANCHORS
        self.separable_conv = separable_conv
        num_coordinate_values = 4
        options = {
            'kernel_size': 3,
            'strides': 1,
            'padding': 'same',
            'bias_initializer': 'zeros',
        }
        if separable_conv:
            kernel_initializer = {
                'depthwise_initializer': initializers.VarianceScaling(),
                'pointwise_initializer': initializers.VarianceScaling(),
            }
            options.update(kernel_initializer)
            self.convs = [layers.SeparableConv2D(filters=width, name=f'{self.name}/box-{i}', **options) for i in range(depth)]
            self.head = layers.SeparableConv2D(filters=self.num_anchors * num_coordinate_values, name=f'{self.name}/box-predict', **options)
        else:
            kernel_initializer = {
                'kernel_initializer': initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
            }
            options.update(kernel_initializer)
            self.convs = [layers.Conv2D(filters=width, name=f'{self.name}/box-{i}', **options) for i in range(depth)]
            self.head = layers.Conv2D(filters=self.num_anchors * num_coordinate_values, name=f'{self.name}/box-predict', **options)
        self.bns = [layers.BatchNormalization(name=f'{self.name}/box-{i}') for i in range(depth)]
        self.relu = layers.Lambda(lambda x: tf.nn.swish(x))
        self.reshape = layers.Reshape((-1, num_coordinate_values))
        self.level = 0

    def call(self, inputs, **kwargs):
        feature = inputs
        for i in range(self.depth):
            feature = self.convs[i](feature)
            feature = self.bns[i](feature)
            feature = self.relu(feature)
        outputs = self.head(feature)
        outputs = self.reshape(outputs)
        return outputs


class ClassNet(models.Model):

    def __init__(self, width, depth, num_classes=20, separable_conv=True, **kwargs):
        super(ClassNet, self).__init__(**kwargs)
        self.width = width
        self.depth = depth
        self.num_classes = num_classes
        self.num_anchors = NUM_ANCHORS
        self.separable_conv = separable_conv
        options = {
            'kernel_size': 3,
            'strides': 1,
            'padding': 'same',
        }
        if self.separable_conv:
            kernel_initializer = {
                'depthwise_initializer': initializers.VarianceScaling(),
                'pointwise_initializer': initializers.VarianceScaling(),
            }
            options.update(kernel_initializer)
            self.convs = [layers.SeparableConv2D(filters=width, bias_initializer='zeros', name=f'{self.name}/class-{i}', **options)
                          for i in range(depth)]
            self.head = layers.SeparableConv2D(filters=num_classes * self.num_anchors,
                                               bias_initializer=PriorProbability(probability=0.01),
                                               name=f'{self.name}/class-predict', **options)
        else:
            kernel_initializer = {
                'kernel_initializer': initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
            }
            options.update(kernel_initializer)
            self.convs = [layers.Conv2D(filters=width, bias_initializer='zeros', name=f'{self.name}/class-{i}', **options)
                          for i in range(depth)]
            self.head = layers.Conv2D(filters=num_classes * self.num_anchors,
                                      bias_initializer=PriorProbability(probability=0.01),
                                      name='class-predict', **options)
        self.bns = [layers.BatchNormalization(name=f'{self.name}/class-{i}') for i in range(depth)]
        self.relu = layers.Lambda(lambda x: tf.nn.swish(x))
        self.reshape = layers.Reshape((-1, num_classes))
        self.activation = layers.Activation('sigmoid')
        self.level = 0

    def call(self, inputs, **kwargs):
        feature = inputs
        for i in range(self.depth):
            feature = self.convs[i](feature)
            feature = self.bns[i](feature)
            feature = self.relu(feature)
        outputs = self.head(feature)
        outputs = self.reshape(outputs)
        outputs = self.activation(outputs)
        return outputs
