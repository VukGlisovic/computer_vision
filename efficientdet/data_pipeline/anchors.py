import tensorflow as tf
from efficientdet.constants import ASPECT_RATIOS, SCALES, NUM_ANCHORS


class AnchorBox:
    """Class that produces anchor boxes of format [x, y, width, height].

    This class is inspired by https://keras.io/examples/vision/retinanet/

    Attributes:
        aspect_ratios (list[float]): the aspect ratios (width / height) of the anchor
            boxes at each location on the feature map.
        scales (list[float]): the scale of the anchor boxes (box multipliers) at each
            location on the feature map. E.g. the highest resolution feature map will
            have anchor boxes width & height of `32 * {2**0, 2**1/3, 2**2/3}`. This is
            before modifying anchor boxes for aspect ratios.
        _num_anchors (int): the number of anchor boxes at each location on the feature map.
        _sizes (list[float]): the anchor widths & heights for each feature map in the feature
            pyramid. The sizes are w.r.t the original image.
        _strides (list[float]): the strides w.r.t the original image for each feature map in
            the feature pyramid. E.g., moving one pixel in the highest resolution feature map
            corresponds to moving `2 ** 3 = 8` pixels in the original image.
    """

    def __init__(self):
        self.aspect_ratios = ASPECT_RATIOS
        self.scales = SCALES

        self._num_anchors = NUM_ANCHORS
        self._strides = [2 ** i for i in range(3, 8)]
        self._sizes = [32.0, 64.0, 128.0, 256.0, 512.0]
        self._anchor_shapes = self._compute_shapes()

    def _compute_shapes(self):
        """Computes anchor box shapes for all sizes, ratios and scales.
        NOTE: these are anchor shapes; thus not actual anchor boxes.
        These shapes are still agnostic of anchor location and thus
        still need to be tiled over the feature maps.

        The output tensor has dimensions (len(_sizes) * len(ratios) * len(scales), 2).
        The output basically is (nr. anchor shapes, [width, height]).

        Returns:
            tf.Tensor
        """
        anchor_dims_all = []
        for size in self._sizes:
            anchor_dims = []
            for ratio in self.aspect_ratios:
                # use sqrt to preserve area size
                anchor_height = size / tf.sqrt(ratio)
                anchor_width = size * tf.sqrt(ratio)
                dims = tf.stack([anchor_width, anchor_height], axis=-1)
                dims = tf.reshape(dims, [1, 1, 2])
                # append different scales of the boxes
                anchor_dims += [scale * dims for scale in self.scales]
            anchor_dims_all.append(tf.stack(anchor_dims, axis=-2))
        return anchor_dims_all

    def _tile_anchors(self, feature_height, feature_width, level):
        """Generates anchor boxes by tiling anchor box shapes over a
        feature map. This is done by combining the [width, height] anchor
        shapes with [center x, center y] coordinates.

        Each feature map pixel is associated with various anchor shapes.
        The number of anchors per pixel is len(ratios) * len(scales).

        The anchor boxes have coordinates w.r.t the input image.
        Each row in the output contains [center x, center y, width, height].
        This function outputs a tensor of shape (feature_height * feature_width * num_anchors_shapes, 4).

        Args:
            feature_height (int): the height of the feature map.
            feature_width (int): the width of the feature map.
            level (int): the level of the feature map in the feature pyramid.

        Returns:
            tf.Tensor: Anchor boxes expressed in input image coordinates
        """
        range_x = tf.range(feature_width, dtype=tf.float32) + 0.5
        range_y = tf.range(feature_height, dtype=tf.float32) + 0.5
        centers = tf.stack(tf.meshgrid(range_x, range_y), axis=-1) * self._strides[level - 3]
        centers = tf.expand_dims(centers, axis=-2)
        centers = tf.tile(centers, [1, 1, self._num_anchors, 1])
        shapes = tf.tile(self._anchor_shapes[level - 3], [feature_height, feature_width, 1, 1])
        anchors = tf.concat([centers, shapes], axis=-1)
        anchors = tf.reshape(anchors, [feature_height * feature_width * self._num_anchors, 4])
        return anchors

    def get_all_anchors(self, image_height, image_width):
        """Generates all anchor boxes by tiling anchor shapes over all
        feature maps of the feature pyramid.

        Args:
            image_height (int):
            image_width (int):

        Returns:
            tf.Tensor: all anchor boxes with shape (total_anchors, 4) where each
                box is encoded as [center x, center y, width, height].
        """
        anchors = [self._tile_anchors(tf.math.ceil(image_height / 2 ** i), tf.math.ceil(image_width / 2 ** i), i)
                   for i in range(3, 8)]
        anchors = tf.concat(anchors, axis=0)
        return anchors
