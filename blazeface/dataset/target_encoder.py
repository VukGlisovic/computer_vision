import tensorflow as tf

from blazeface.dataset import utils
from blazeface.constants import *


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
    bbox_y1, bbox_x1, bbox_y2, bbox_x2 = tf.split(bboxes, 4, axis=-1)
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


def calculate_bboxes_and_landmarks_deltas(anchor_boxes, bboxes_and_landmarks):
    """Calculating bounding box and landmark deltas for given ground truth boxes and landmarks.

    Args:
        anchor_boxes (tf.Tensor): shape [n_anchors, 4] with coordinates [center_x, center_y, width, height].
        bboxes_and_landmarks (tf.Tensor): shape [bs, n_anchors, 4 + N_LANDMARKS * 2] where we have
            coordinates [x1, y1, x2, y2, lmark_x0, lmark_y0, ..., lmark_xN, lmark_yN]

    Returns:
        tf.Tensor: target deltas of shape [bs, n_boxes, 4 + N_LANDMARKS * 2]
    """
    # convert ground truth coordinates to [x, y, w, h] notation
    gt_width = bboxes_and_landmarks[..., 2] - bboxes_and_landmarks[..., 0]
    gt_height = bboxes_and_landmarks[..., 3] - bboxes_and_landmarks[..., 1]
    gt_center_x = bboxes_and_landmarks[..., 0] + 0.5 * gt_width
    gt_center_y = bboxes_and_landmarks[..., 1] + 0.5 * gt_height
    # calculate difference between ground truth and anchor box coordinates
    delta_x = (gt_center_x - anchor_boxes[..., 0]) / anchor_boxes[..., 2]
    delta_y = (gt_center_y - anchor_boxes[..., 1]) / anchor_boxes[..., 3]
    delta_w = gt_width / anchor_boxes[..., 2]
    delta_h = gt_height / anchor_boxes[..., 3]
    # calculate difference between landmark locations and anchor centers
    total_landmarks = tf.shape(bboxes_and_landmarks[..., 4:])[-1] // 2
    xy_pairs = tf.tile(anchor_boxes[..., 0:2], (1, total_landmarks))
    wh_pairs = tf.tile(anchor_boxes[..., 2:4], (1, total_landmarks))
    landmark_deltas = (bboxes_and_landmarks[..., 4:] - xy_pairs) / wh_pairs

    bbox_deltas = tf.stack([delta_x, delta_y, delta_w, delta_h], axis=-1)
    return tf.concat([bbox_deltas, landmark_deltas], axis=-1)


def calculate_targets(anchor_boxes, gt_boxes, gt_landmarks):
    """Calculates the delta targets between the anchor boxes and the
    ground truth boxes.

    Args:
        anchor_boxes (tf.Tensor): shape [n_anchors, 4] where we have
            [center_x, center_y, width, height] values in normalized
            format between [0, 1].
        gt_boxes (tf.Tensor): shape [bs, n_bboxes, 4) where we have
            [xmin, ymin, xmax, ymax] values in normalized format
            between [0, 1].
        gt_landmarks (tf.Tensor): shape [bs, n_bboxes, total_landmarks, 2)
            where we have [x, y] locations of the landmarks in normalized
            format between [0, 1].

    Returns:
        tuple[tf.Tensor, tf.Tensor]: where we have deltas of shape [bs, n_anchors, 4 + N_LANDMARKS * 2]
            and we have the labels of shape [bs, n_anchors, 1]. The deltas contain [delta_center_x, delta_center_y, delta_w, delta_h, delta_lmark_x0, delta_lmark_y0, ..., delta_lmark_xN, delta_lmark_yN]
            and the labels have values 0 or 1.
    """
    batch_size = tf.shape(gt_boxes)[0]
    # calculate IOU values between ground truth boxes and bounding boxes
    iou_map = generate_iou_map(gt_boxes, utils.xywh_to_xyxy(anchor_boxes))
    # get index of best matching ground truth box for each anchor
    idx_best_match_gt_box = tf.argmax(iou_map, axis=2, output_type=tf.int32)
    # get corresponding IOU value for the best matches
    iou_best_match_gt_box = tf.reduce_max(iou_map, axis=2)
    # find out which anchors have a positive match with the ground truth boxes
    pos_cond = tf.greater(iou_best_match_gt_box, IOU_THRESHOLD)
    # concat bounding boxes with landmarks
    gt_landmarks = tf.reshape(gt_landmarks, (batch_size, -1, N_LANDMARKS * 2))  # merge landmarks and coordinates dimensions
    gt_boxes_and_landmarks = tf.concat([gt_boxes, gt_landmarks], axis=-1)
    # gather the best matched ground truth box to each anchor
    gt_boxes_and_landmarks_map = tf.gather(gt_boxes_and_landmarks, idx_best_match_gt_box, batch_dims=1)
    gt_boxes_and_landmarks_matches = tf.where(tf.expand_dims(pos_cond, axis=-1), gt_boxes_and_landmarks_map, tf.zeros_like(gt_boxes_and_landmarks_map))
    target_deltas = calculate_bboxes_and_landmarks_deltas(anchor_boxes, gt_boxes_and_landmarks_matches) / (BOX_VARIANCE + LMARK_VARIANCE)
    # convert positive IOU matches tensor to labels
    target_labels = tf.expand_dims(tf.cast(pos_cond, dtype=tf.float32), axis=-1)

    return target_deltas, target_labels
