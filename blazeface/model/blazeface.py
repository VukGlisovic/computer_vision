import tensorflow as tf

from blazeface.model.custom_blocks import BlazeBlock, DoubleBlazeBlock


class BlazeFaceModel(tf.keras.models.Model):
    """
    BlazeBlock as described in
    "BlazeFace: Sub-millisecond Neural Face Detection on Mobile GPUs" (https://arxiv.org/pdf/1907.05047.pdf)

    Meaning a higher receptive field is used by applying a kernel size of 5x5.

    Args:
        detections_per_layer (list): whether to downsample or not.
        total_reg_points (int): number of output feature maps.
    """

    def __init__(self, detections_per_layer, total_reg_points):
        super(BlazeFaceModel, self).__init__()
        total_reg_points = 6 * 2 + 4  # hyper_params["total_landmarks"] * 2 + 4
        # input conv
        self.input_conv = tf.keras.layers.Conv2D(filters=24, kernel_size=(5, 5), strides=2, padding="same", activation="relu")
        # 5 blaze blocks
        self.blaze_blocks = []
        for stride, filters in [(1, 24), (1, 24), (2, 48), (1, 48), (1, 48)]:
            self.blaze_blocks.append(BlazeBlock(stride, filters))
        # 6 double blaze blocks
        self.double_blaze_blocks_part1 = []
        for stride, filters1, filters2 in [(2, 24, 96), (1, 24, 96), (1, 24, 96)]:
            self.double_blaze_blocks_part1.append(DoubleBlazeBlock(stride, filters1, filters2))
        self.double_blaze_blocks_part2 = []
        for stride, filters1, filters2 in [(2, 24, 96), (1, 24, 96), (1, 24, 96)]:
            self.double_blaze_blocks_part2.append(DoubleBlazeBlock(stride, filters1, filters2))

        self.dbb3_to_labels = tf.keras.layers.Conv2D(detections_per_layer[0], kernel_size=(3, 3), padding="same")
        self.dbb6_to_labels = tf.keras.layers.Conv2D(detections_per_layer[1], kernel_size=(3, 3), padding="same")

        self.dbb3_to_boxes = tf.keras.layers.Conv2D(detections_per_layer[0] * total_reg_points, kernel_size=(3, 3), padding="same")
        self.dbb6_to_boxes = tf.keras.layers.Conv2D(detections_per_layer[1] * total_reg_points, kernel_size=(3, 3), padding="same")

        self.labels_reshape = tf.keras.layers.Reshape((-1, 1))
        self.labels_concat = tf.keras.layers.Concatenate(axis=1)
        self.labels_act = tf.keras.layers.Activation('sigmoid')

        self.boxes_reshape = tf.keras.layers.Reshape((-1, total_reg_points))
        self.boxes_concat = tf.keras.layers.Concatenate(axis=1)

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

        dbb3_out_labels = self.dbb3_to_labels(dbb3_out)
        dbb6_out_labels = self.dbb6_to_labels(dbb6_out)

        dbb3_out_boxes = self.dbb3_to_boxes(dbb3_out)
        dbb6_out_boxes = self.dbb6_to_boxes(dbb6_out)

        dbb3_out_labels = self.labels_reshape(dbb3_out_labels)
        dbb6_out_labels = self.labels_reshape(dbb6_out_labels)
        out_labels = self.labels_concat([dbb3_out_labels, dbb6_out_labels])
        out_labels = self.labels_act(out_labels)

        dbb3_out_boxes = self.boxes_reshape(dbb3_out_boxes)
        dbb6_out_boxes = self.boxes_reshape(dbb6_out_boxes)
        out_boxes = self.boxes_concat([dbb3_out_boxes, dbb6_out_boxes])

        return [out_labels, out_boxes, dbb3_out_labels, dbb6_out_labels, dbb3_out_boxes, dbb6_out_boxes]
