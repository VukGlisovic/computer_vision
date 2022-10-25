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


def adjust_bbox_coordinates(bboxes, min_max):
    """Adjust bounding boxes coordinates to match the new boundaries of
    the image.

    Args:
        bboxes (tf.Tensor):
        min_max (tf.Tensor): new min and max coordinates relative to
            the original image (before adjustments).

    Returns:
        tf.Tensor: adjusted bounding box coordinates
    """
    y_min, x_min, y_max, x_max = tf.split(min_max, 4)
    renomalized_bboxes = bboxes - tf.concat([x_min, y_min, x_min, y_min], axis=-1)
    renomalized_bboxes /= tf.concat([x_max-x_min, y_max-y_min, x_max-x_min, y_max-y_min], axis=-1)
    return tf.clip_by_value(renomalized_bboxes, 0, 1)


def adjust_landmark_coordinates(landmarks, min_max):
    """Adjust landmark coordinates to match the new boundaries of
    the image.

    Args:
        landmarks (tf.Tensor):
        min_max (tf.Tensor): new min and max coordinates relative to
            the original image (before adjustments).

    Returns:
        tf.Tensor: adjusted landmark coordinates
    """
    y_min, x_min, y_max, x_max = tf.split(min_max, 4)
    renomalized_landmarks = landmarks - tf.concat([y_min, x_min], -1)
    renomalized_landmarks /= tf.concat([y_max-y_min, x_max-x_min], -1)
    return tf.clip_by_value(renomalized_landmarks, 0, 1)


def denormalize_bboxes(bboxes, width, height):
    """Change relative bounding boxes coordinates to absolute
    coordinates in the image.

    Args:
        bboxes (tf.Tensor): shape [..., [x1, y1, x2, y2]) in the [0, 1] range.
        height (int): image height to adjust the coordinates to
        width (int): image width to adjust the coordinates to

    Returns:
        tf.Tensor: shape [..., [x1, y1, x2, y2]]
    """
    x1 = bboxes[..., 0] * width
    y1 = bboxes[..., 1] * height
    x2 = bboxes[..., 2] * width
    y2 = bboxes[..., 3] * height
    return tf.round(tf.stack([x1, y1, x2, y2], axis=-1))


def denormalize_landmarks(lmarks, width, height):
    """Change relative landmark coordinates to absolute
    coordinates in the image.

    Args:
        lmarks (tf.Tensor): shape [..., [x, y]] in the [0, 1] range.
        height (int): image height to adjust the coordinates to
        width (int): image width to adjust the coordinates to

    Returns:
        tf.Tensor: shape [..., [x, y]]
    """
    return tf.round(lmarks * tf.cast([width, height], tf.float32))


def generate_iou_map(gt_boxes, bboxes, bboxes_perm=[0, 2, 1]):
    """Computes the pairwise IOU matrix for two sets of boxes. All boxes
    should be of format [xmin, ymin, xmax, ymax].

    Args:
        gt_boxes (tf.Tensor): shape [n_anchors, 4]
        bboxes (tf.Tensor): shape [bs, n_bboxes, 4]. Here n_bboxes will usually
            be 1 since the model is meant to be trained on a single face.
        bboxes_perm (list): transposes the bboxes such that the coordinates are
            of the gt_boxes and the bboxes are aligned.

    Returns:
        tf.Tensor: shape [bs, n_anchors, n_bboxes]
    """
    gt_rank = tf.rank(gt_boxes)
    gt_expand_axis = gt_rank - 2
    # get ground truth and anchor coordinates
    gt_x1, gt_y1, gt_x2, gt_y2 = tf.split(gt_boxes, 4, axis=-1)
    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = tf.split(bboxes, 4, axis=-1)
    # calculate ground truth box and bounding box areas
    gt_area = tf.squeeze((gt_y2 - gt_y1) * (gt_x2 - gt_x1), axis=-1)
    bbox_area = tf.squeeze((bbox_y2 - bbox_y1) * (bbox_x2 - bbox_x1), axis=-1)
    # calculate intersection area
    x_top = tf.maximum(bbox_x1, tf.transpose(gt_x1, bboxes_perm))
    y_top = tf.maximum(bbox_y1, tf.transpose(gt_y1, bboxes_perm))
    x_bottom = tf.minimum(bbox_x2, tf.transpose(gt_x2, bboxes_perm))
    y_bottom = tf.minimum(bbox_y2, tf.transpose(gt_y2, bboxes_perm))
    intersection_area = tf.maximum(x_bottom - x_top, 0) * tf.maximum(y_bottom - y_top, 0)  # if no overlap between boxes, take 0
    # calculate union area
    union_area = (tf.expand_dims(bbox_area, -1) + tf.expand_dims(gt_area, gt_expand_axis) - intersection_area)

    return intersection_area / union_area
