import tensorflow as tf
from efficientdet.constants import BOX_VARIANCE
from efficientdet.data_pipeline.utils import  xywh_to_xyxy
from efficientdet.data_pipeline.anchors import AnchorBox


def pairwise_iou(boxes1, boxes2):
    """Computes the pairwise IOU matrix for two sets of boxes. All boxes
    should be of format [center x, center y, width, height].

    Args:
        boxes1 (tf.Tensor): shape (N, 4)
        boxes2 (tf.Tensor): shape (M, 4)

    Returns:
        tf.Tensor: shape (N, M), where index (i, j) is the IOU between boxes1[i] and boxes2[j]
    """
    boxes1_xyxy = xywh_to_xyxy(boxes1)
    boxes2_xyxy = xywh_to_xyxy(boxes2)
    left_top = tf.maximum(boxes1_xyxy[:, None, :2], boxes2_xyxy[:, :2])
    right_bot = tf.minimum(boxes1_xyxy[:, None, 2:], boxes2_xyxy[:, 2:])
    intersection = tf.maximum(0.0, right_bot - left_top)
    intersection_area = intersection[:, :, 0] * intersection[:, :, 1]
    boxes1_area = boxes1[:, 2] * boxes1[:, 3]
    boxes2_area = boxes2[:, 2] * boxes2[:, 3]
    union_area = tf.maximum(boxes1_area[:, None] + boxes2_area - intersection_area, 1e-8)
    iou_matrix = tf.clip_by_value(intersection_area / union_area, 0.0, 1.0)
    return iou_matrix


class TargetEncoder:
    """ Transforms raw box coordinates and class integers into
        dense anchor classification and regression targets for training.

        Note that each layer of the neck a.k.a feature pyramid network is
        associated with differently sized anchors. Therefore, each layer
        gets its own classification and regression targets.

    Attributes:
        _anchor_box: Anchor box generator.
        _box_variance: These are normalization values. They scale bounding box targets
            with the goal of having standard normally distributed targets. More details
            can be found at https://leimao.github.io/blog/Bounding-Box-Encoding-Decoding/
    """

    def __init__(self):
        self._anchor_box = AnchorBox()
        self._box_variance = BOX_VARIANCE

    @staticmethod
    def _match_anchor_boxes(anchor_boxes, gt_boxes, positive_iou=0.5, negative_iou=0.4):
        """Matches ground truth boxes to anchor boxes based on IOU.
        For each anchor box, we check whether we want to use it as a positive
        anchor (ground truth example), a negative anchor (background class) or
        ignore it.
        iou >= positive_iou: positive anchor
        iou < negative_iou: negative anchor
        positive_iou > iou >= negative_iou: ignore anchor
        This method returns the index of ground truths that best matched an anchor,
        a mask for positive anchor boxes and a mask for anchor boxes that should be
        ignored.

        Args:
            anchor_boxes (tf.Tensor): shape (total_anchors, 4) where each row is [x, y, width, height]
            gt_boxes (tf.Tensor): shape (num_objects, 4) where each row is [x, y, width, height]
            positive_iou (float): minimum IOU threshold for positive anchors
            negative_iou (float): maximum IOU threshold for negative anchors

        Returns:
            tuple[tf.Tensor, tf.Tensor, tf.Tensor]
        """
        iou_matrix = pairwise_iou(anchor_boxes, gt_boxes)
        max_iou_per_anchor = tf.reduce_max(iou_matrix, axis=1)
        matched_gt_idx = tf.argmax(iou_matrix, axis=1)
        positive_mask = tf.greater_equal(max_iou_per_anchor, positive_iou)
        negative_mask = tf.less(max_iou_per_anchor, negative_iou)
        ignore_mask = tf.logical_not(tf.logical_or(positive_mask, negative_mask))
        positive_mask = tf.cast(positive_mask, dtype=tf.float32)
        ignore_mask = tf.cast(ignore_mask, dtype=tf.float32)

        return matched_gt_idx, positive_mask, ignore_mask

    def _compute_regression_target(self, anchor_boxes, matched_gt_boxes):
        """Creates targets for the box regressions; the amount by which
        the anchors must be adjusted in center x, center y, width and height
        in order to perfectly match the ground truth boxes.

        These targets are used to train the model's regression head. For an
        explanation of these encodings, see Section 3.1.2 of Faster RCNN paper:
        https://arxiv.org/pdf/1506.01497.pdf

        The encodings are as follows:
        [
          (x center gt - x center anchor) / anchor width,
          (y center gt - y center anchor) / anchor height,
          log(width gt / width anchor),
          log(height gt / height anchor)
        ]

        One can interpret these encodings as generalizing across anchor shapes;
        Adjustments in the x coordinate are relative to the size of the anchor
        box, as larger boxes will require larger center coordinate refinements.

        This method outputs an encoded box refinement target.

        Args:
            anchor_boxes (tf.Tensor): shape (total_anchors, 4) with rows [center x, center y, width, height]
            matched_gt_boxes (tf.Tensor): each anchor is associated with a matched ground truth box (regardless
                of whether this match is a positive match or not to produce a training signal). This function
                encodes the difference between the anchor and the best matched ground truth.

        Returns:
            tf.Tensor
        """
        box_target = tf.concat(
            [
                (matched_gt_boxes[:, :2] - anchor_boxes[:, :2]) / anchor_boxes[:, 2:],
                tf.math.log(matched_gt_boxes[:, 2:] / anchor_boxes[:, 2:]),
            ],
            axis=-1,
        )
        box_target = box_target / self._box_variance
        return box_target

    def encode_sample(self, img, boxes, labels):
        """Creates regression and classification targets for one sample. It produces
        box refinement and classification targets.
        -1 indicates negative anchors
        -2 indicates anchors that need to be ignored

        Args:
            img (tf.Tensor): only used to obtain the image resolution
            boxes (tf.Tensor): ground truth boxes, of format [x center, y center, width, height]
            labels (tf.Tensor): ground truth class integers.

        Returns:
            tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        image_shape = img.get_shape()
        anchor_boxes = self._anchor_box.get_all_anchors(image_shape[0], image_shape[1])
        matched_gt_idx, positive_mask, ignore_mask = self._match_anchor_boxes(anchor_boxes, boxes)
        matched_gt_boxes = tf.gather(boxes, matched_gt_idx)
        box_target = self._compute_regression_target(anchor_boxes, matched_gt_boxes)

        labels = tf.cast(labels, dtype=tf.float32)
        matched_gt_cls_ids = tf.gather(labels, matched_gt_idx)
        cls_target = tf.where(tf.not_equal(positive_mask, 1.0), -1.0, matched_gt_cls_ids)
        cls_target = tf.where(tf.equal(ignore_mask, 1.0), -2.0, cls_target)
        # cls_target = tf.expand_dims(cls_target, axis=-1)

        positive_mask = tf.expand_dims(tf.cast(positive_mask, tf.float32), axis=-1)
        box_target = tf.concat([box_target, positive_mask], axis=-1)
        return img, box_target, cls_target
