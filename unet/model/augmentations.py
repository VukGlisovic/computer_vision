import tensorflow as tf


def rotate(img, mask):
    """Rotates the input image and mask by 0, 90, 180 or 270 degrees.

    Args:
        img (tf.Tensor):

    Returns:
        tf.Tensor
    """
    nr_rotations = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.dtypes.int32)
    return tf.image.rot90(img, nr_rotations), tf.image.rot90(mask, nr_rotations)


def flip(img, mask):
    """Takes the input image and the target mask and applies the
    same flipping with a 50/50 chance. Thus a 50% chance of flipping
    left and right and then another 50% chance of flipping top and
    bottom.

    Args:
        img (tf.Tensor):
        mask (tf.Tensor):

    Returns:
        tuple[tf.Tensor, tf.Tensor]
    """
    img, mask = tf.cond(
        tf.random.uniform(shape=[], minval=0, maxval=2, dtype=tf.dtypes.int32) == 1,
        true_fn=lambda: (tf.reverse(img, axis=[0]), tf.reverse(mask, axis=[0])),
        false_fn=lambda: (img, mask)
    )
    img, mask = tf.cond(
        tf.random.uniform(shape=[], minval=0, maxval=2, dtype=tf.dtypes.int32) == 1,
        true_fn=lambda: (tf.reverse(img, axis=[1]), tf.reverse(mask, axis=[1])),
        false_fn=lambda: (img, mask)
    )
    return img, mask
