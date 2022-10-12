import tensorflow as tf


def xyxy_to_xywh(boxes):
    """Transforms [xmin, ymin, xmax, ymax] to [center x, center y, width, height].
    Boxes can both be batched or not.

    Args:
        boxes (tf.Tensor):

    Returns:
        tf.Tensor
    """
    boxes = tf.cast(boxes, dtype=tf.float32)
    boxes_transformed = tf.concat(
        [(boxes[..., :2] + boxes[..., 2:]) / 2.0, boxes[..., 2:] - boxes[..., :2]],
        axis=-1
    )
    return boxes_transformed


def xywh_to_xyxy(boxes):
    """Transforms [center x, center y, width, height] to [xmin, ymin, xmax, ymax].
    Boxes can both be batched or not.

    Args:
      boxes (tf.Tensor):

    Returns:
        tf.Tensor
    """
    boxes_transformed = tf.concat(
        [boxes[..., :2] - boxes[..., 2:] / 2.0, boxes[..., :2] + boxes[..., 2:] / 2.0],
        axis=-1
    )
    return boxes_transformed
