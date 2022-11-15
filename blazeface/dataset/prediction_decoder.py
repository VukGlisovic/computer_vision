import tensorflow as tf

from blazeface.constants import *
from blazeface.dataset import utils


def get_bboxes_and_landmarks_from_deltas(anchor_boxes, deltas):
    """Given the anchor boxes, it decodes the deltas to prediction coordinates
    for the bounding boxes and the landmarks.

    Args:
        anchor_boxes (tf.Tensor): shape [n_anchors, 4] with [center_x, center_y, width, height].
        deltas (tf.Tensor): shape [bs, n_anchors, N_deltas] with [delta_center_x, delta_center_y, delta_w, delta_h, delta_lmark_x0, delta_lmark_y0, ..., delta_lmark_xN, delta_lmark_yN]

    Returns:
        tf.Tensor: bboxes and landmarks with shape [bs, n_anchors, [x1, y1, x2, y2, lmark_x0, lmark_y0, ..., lmark_xN, lmark_yN]]
    """
    deltas = deltas * (BOX_VARIANCE + LMARK_VARIANCE)
    # convert deltas to xywh coordinates
    bbox_center_x = (deltas[..., 0] * anchor_boxes[..., 2]) + anchor_boxes[..., 0]
    bbox_center_y = (deltas[..., 1] * anchor_boxes[..., 3]) + anchor_boxes[..., 1]
    bbox_width = deltas[..., 2] * anchor_boxes[..., 2]
    bbox_height = deltas[..., 3] * anchor_boxes[..., 3]
    # decode further from xywh to xyxy
    x1 = bbox_center_x - (0.5 * bbox_width)
    y1 = bbox_center_y - (0.5 * bbox_height)
    x2 = x1 + bbox_width
    y2 = y1 + bbox_height
    # decode landmark deltas to xy coordinates
    xy_pairs = tf.tile(anchor_boxes[..., 0:2], (1, N_LANDMARKS))
    wh_pairs = tf.tile(anchor_boxes[..., 2:4], (1, N_LANDMARKS))
    landmarks = (deltas[..., 4:] * wh_pairs) + xy_pairs
    # concatenate bbox and landmark coordinates
    coordinates = tf.concat([tf.stack([x1, y1, x2, y2], axis=-1), landmarks], -1)
    return tf.clip_by_value(coordinates, 0, 1)


def get_weighted_boxes_and_landmarks(scores, bboxes_and_landmarks, mask):
    """Calculates a weighted mean of given bboxes and landmarks according to the mask.

    Args:
        scores (tf.Tensor): shape [n_anchors, 1] containing probabilities
        bboxes_and_landmarks (tf.Tensor): shape [n_anchors, [x1, y1, x2, y2, landmark_x0, landmark_y0, ..., landmark_xN, landmark_yN]]
        mask (tf.Tensor): shape [n_anchors,]

    Returns:
        tf.Tensor: weighted_bbox_and_landmark with shape [1, [x1, y1, x2, y2, lmark_x0, lmark_y0, ..., lmark_xN, lmark_yN]]
    """
    selected_scores = scores[mask]
    selected_bboxes_and_landmarks = bboxes_and_landmarks[mask]
    weighted_sum = tf.reduce_sum(selected_bboxes_and_landmarks * selected_scores, 0)
    sum_selected_scores = tf.reduce_sum(selected_scores, 0)
    sum_selected_scores = tf.where(tf.equal(sum_selected_scores, 0.0), 1.0, sum_selected_scores)
    return tf.expand_dims(weighted_sum / sum_selected_scores, 0)


def weighted_suppression_body(counter, iou_threshold, scores, bboxes_and_landmarks, weighted_suppressed_data):
    """Weighted mean suppression algorithm while body.

    Args:
        counter (tf.Tensor): while body counter
        iou_threshold (float): threshold value for overlapping bounding boxes
        scores (tf.Tensor): shape [n_anchors, 1] containing probabilities
        bboxes_and_landmarks (tf.Tensor): shape [n_anchors, [x1, y1, x2, y2, lmark_x0, lmark_y0, ..., lmark_xN, lmark_yN]]
        weighted_suppressed_data (tf.Tensor): shape [M, [x1, y1, x2, y2, lmark_x0, lmark_y0, ..., lmark_xN, lmark_yN]]

    Returns:
        tuple[tf.Tensor, float, tf.Tensor, tf.Tensor, tf.Tensor]: containing the following
            counter
            iou_threshold
            scores with shape [n_anchors - N, probability]
            bboxes_and_landmarks with shape [n_anchors - N, [x1, y1, x2, y2, lmark_x0, lmark_y0, ..., lmark_xN, lmark_yN]]
            weighted_suppressed_data with shape [M + 1, [x1, y1, x2, y2, lmark_x0, lmark_y0, ..., lmark_xN, lmark_yN]]
    """
    counter = tf.add(counter, 1)
    first_box = bboxes_and_landmarks[0, 0:4]
    iou_map = utils.generate_iou_map(bboxes_and_landmarks[..., 0:4], first_box, bboxes_perm=[1, 0])
    overlapped_mask = tf.reshape(tf.greater(iou_map, iou_threshold), (-1,))
    weighted_bbox_and_landmark = get_weighted_boxes_and_landmarks(scores, bboxes_and_landmarks, overlapped_mask)
    weighted_suppressed_data = tf.concat([weighted_suppressed_data, weighted_bbox_and_landmark], axis=0)
    not_overlapped_mask = tf.logical_not(overlapped_mask)
    scores = scores[not_overlapped_mask]
    bboxes_and_landmarks = bboxes_and_landmarks[not_overlapped_mask]
    return counter, iou_threshold, scores, bboxes_and_landmarks, weighted_suppressed_data


def weighted_suppression(scores, bboxes_and_landmarks, max_total_size=50, score_threshold=0.75, iou_threshold=0.3):
    """Blazeface's weighted mean suppression algorithm.

    Args:
        scores (tf.Tensor): shape [n_anchors, 1] containing prediction scores
        bboxes_and_landmarks (tf.Tensor): shape [n_anchors, N_deltas] containing [x1, y1, x2, y2, lmark_x0, lmark_y0, ..., lmark_xN, lmark_yN]
        max_total_size (int): maximum returned bounding boxes and landmarks
        score_threshold (float): threshold value for bounding boxes and landmarks selection
        iou_threshold (float): threshold value for overlapping bounding boxes

    Returns:
        tf.Tensor: weighted_bboxes_and_landmarks = (dynamic_size, [y1, x1, y2, x2, landmark_x0, landmark_y0, ..., landmark_xN, landmark_yN])
    """
    score_mask = tf.squeeze(tf.greater(scores, score_threshold), -1)
    scores = scores[score_mask]
    bboxes_and_landmarks = bboxes_and_landmarks[score_mask]
    sorted_indices = tf.argsort(scores, axis=0, direction="DESCENDING")
    sorted_scores = tf.gather_nd(scores, sorted_indices)
    sorted_bboxes_and_landmarks = tf.gather_nd(bboxes_and_landmarks, sorted_indices)
    counter = tf.constant(0, tf.int32)
    weighted_data = tf.zeros(tf.shape(bboxes_and_landmarks[0: 1]), dtype=tf.float32)
    cond = lambda counter, iou_threshold, scores, data, weighted: tf.logical_and(tf.less(counter, max_total_size), tf.greater(tf.shape(scores)[0], 0))
    _, _, _, _, weighted_data = tf.while_loop(cond, weighted_suppression_body, [counter, iou_threshold, sorted_scores, sorted_bboxes_and_landmarks, weighted_data])
    weighted_data = weighted_data[1:]
    pad_size = max_total_size - weighted_data.shape[0]
    weighted_data = tf.pad(weighted_data, ((0, pad_size), (0, 0)))
    return weighted_data
