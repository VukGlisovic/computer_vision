import tensorflow as tf

from blazeface.constants import FMAPS_WITH_ANCHORS, N_ANCHORS_PER_LOC


def get_anchor_scale(m, n=4, scale_min=0.1484375, scale_max=0.75):
    """Get the scale for a particular a certain index.

    Based on: "SSD: Single Shot MultiBox Detector" section 2.2.
    https://arxiv.org/pdf/1512.02325.pdf

    Args:
        m (int):
        n (int):
        scale_min (float):
        scale_max (float):

    Returns:
        float
    """
    return scale_min + ((scale_max - scale_min) / (n - 1)) * (m - 1)


def generate_anchors():
    """Creates all the anchors for all feature map locations. It
    returns all anchors in normalized format; [0,1] range.
    Note that it only creates square anchors as described by the
    blazeface paper:
    "Due to the limited variance in human face aspect ratios, limiting
    the anchors to the 1:1 aspect ratio was found sufficient for
    accurate face detection."

    Returns:
        tf.Tensor
    """
    anchors = []
    idx = 0
    for fmap_size, n_anchors in zip(FMAPS_WITH_ANCHORS, N_ANCHORS_PER_LOC):
        for i in range(n_anchors // 2):
            current_scale = get_anchor_scale(idx + 1)
            next_scale = get_anchor_scale(idx + 2)
            # create anchor centers coordinates
            grid_coords = tf.cast((tf.range(0, fmap_size, dtype=tf.float32) + 0.5) / fmap_size, dtype=tf.float32)
            grid_x, grid_y = tf.meshgrid(grid_coords, grid_coords)
            center_x, center_y = tf.reshape(grid_x, (-1, 1)), tf.reshape(grid_y, (-1, 1))
            centers = tf.concat((center_x, center_y), axis=1)
            centers = tf.stack([centers, centers], axis=1)
            centers = tf.reshape(centers, shape=[-1, 2])
            # blazeface only has square anchors
            height = current_scale  # current_scale / tf.sqrt(aspect_ratio) = current_scale / tf.sqrt(1)
            width = current_scale  # current_scale * tf.sqrt(aspect_ratio) = current_scale * tf.sqrt(1)
            next_height = next_width = tf.sqrt(current_scale * next_scale)
            anchor_shapes = tf.cast([[width, height],
                                     [next_width, next_height]], dtype=tf.float32)
            anchor_shapes = tf.tile(anchor_shapes, [fmap_size * fmap_size, 1])
            # concat centers with anchors
            anchors.append(tf.concat((centers, anchor_shapes), axis=1))
            # go to next scale
            idx += 1

    anchors = tf.concat(anchors, axis=0)
    return tf.clip_by_value(anchors, 0, 1)
