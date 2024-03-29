import logging
import tensorflow as tf

from blazeface.model.custom_blocks import BlazeBlock, DoubleBlazeBlock
from blazeface.dataset import anchors
from blazeface.constants import N_ANCHORS_PER_LOC, N_LANDMARKS, IMG_SIZE


class BlazeFaceModel(tf.keras.models.Model):
    """
    BlazeBlock as described in
    "BlazeFace: Sub-millisecond Neural Face Detection on Mobile GPUs" (https://arxiv.org/pdf/1907.05047.pdf)

    Meaning a higher receptive field is used by applying a kernel size of 5x5.

    The feature map shapes are based on an input image of shape 128x128.
    """

    def __init__(self):
        super(BlazeFaceModel, self).__init__()
        # input conv (in [bs, 128, 128, 3], out [bs, 64, 64, 24])
        self.input_conv = tf.keras.layers.Conv2D(filters=24, kernel_size=(5, 5), strides=2, padding="same", activation="relu")
        # 5 blaze blocks (out [bs, 32, 32, 48])
        self.blaze_blocks = []
        for stride, filters in [(1, 24), (1, 24), (2, 48), (1, 48), (1, 48)]:
            self.blaze_blocks.append(BlazeBlock(stride, filters))
        # 6 double blaze blocks
        self.double_blaze_blocks_part1 = []  # (out [bs, 16, 16, 96])
        for stride, filters1, filters2 in [(2, 24, 96), (1, 24, 96), (1, 24, 96)]:
            self.double_blaze_blocks_part1.append(DoubleBlazeBlock(stride, filters1, filters2))
        self.double_blaze_blocks_part2 = []  # (out [bs, 8, 8, 96])
        for stride, filters1, filters2 in [(2, 24, 96), (1, 24, 96), (1, 24, 96)]:
            self.double_blaze_blocks_part2.append(DoubleBlazeBlock(stride, filters1, filters2))

        self.dbb3_to_labels = tf.keras.layers.Conv2D(N_ANCHORS_PER_LOC[0], kernel_size=(3, 3), padding="same")
        self.dbb6_to_labels = tf.keras.layers.Conv2D(N_ANCHORS_PER_LOC[1], kernel_size=(3, 3), padding="same")

        total_regression_points = 4 + N_LANDMARKS * 2  # 4 box coordinates and landmark coordinates
        self.dbb3_to_deltas = tf.keras.layers.Conv2D(N_ANCHORS_PER_LOC[0] * total_regression_points, kernel_size=(3, 3), padding="same")  # 16*16*2*total_regression_points=512*total_regression_points output channels
        self.dbb6_to_deltas = tf.keras.layers.Conv2D(N_ANCHORS_PER_LOC[1] * total_regression_points, kernel_size=(3, 3), padding="same")  # 8*8*6*total_regression_points=384*total_regression_points output channels

        self.labels_reshape = tf.keras.layers.Reshape((-1, 1))
        self.labels_concat = tf.keras.layers.Concatenate(axis=1)
        self.labels_act = tf.keras.layers.Activation('sigmoid')

        self.deltas_reshape = tf.keras.layers.Reshape((-1, total_regression_points))
        self.deltas_concat = tf.keras.layers.Concatenate(axis=1)

    def call(self, inputs):
        x = self.input_conv(inputs)
        for bb in self.blaze_blocks:
            x = bb(x)
        for dbb in self.double_blaze_blocks_part1:
            x = dbb(x)
        dbb3_out = x
        for dbb in self.double_blaze_blocks_part2:
            x = dbb(x)
        dbb6_out = x

        # label predictions
        dbb3_out_labels = self.dbb3_to_labels(dbb3_out)
        dbb6_out_labels = self.dbb6_to_labels(dbb6_out)

        dbb3_out_labels = self.labels_reshape(dbb3_out_labels)
        dbb6_out_labels = self.labels_reshape(dbb6_out_labels)
        out_labels = self.labels_concat([dbb3_out_labels, dbb6_out_labels])
        out_labels = self.labels_act(out_labels)

        # bounding box coordinate delta predictions
        dbb3_out_deltas = self.dbb3_to_deltas(dbb3_out)
        dbb6_out_deltas = self.dbb6_to_deltas(dbb6_out)

        dbb3_out_deltas = self.deltas_reshape(dbb3_out_deltas)
        dbb6_out_deltas = self.deltas_reshape(dbb6_out_deltas)
        out_deltas = self.deltas_concat([dbb3_out_deltas, dbb6_out_deltas])

        return {'deltas': out_deltas, 'labels': out_labels}

    def init_model_weights(self):
        """Initializes the model weights by running an inference.
        """
        _ = self(tf.random.normal(shape=(1, IMG_SIZE, IMG_SIZE, 3)))


def load_model(checkpoint_path, summary=False):
    """Creates a BlazeFace model and loads weights from a checkpoint.

    Args:
        checkpoint_path (str):
        summary (bool):

    Returns:
        tuple[tf.keras.models.Model, tf.Tensor]
    """
    model = BlazeFaceModel()
    model.init_model_weights()
    model.load_weights(checkpoint_path)
    if summary:
        model.summary()
    logging.info("Loaded model from checkpoint: %s", checkpoint_path)
    all_anchors = anchors.generate_anchors()
    logging.info("Created all anchors.")
    return model, all_anchors
