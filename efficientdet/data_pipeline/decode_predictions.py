import tensorflow as tf

from efficientdet.constants import BOX_VARIANCE
from efficientdet.data_pipeline.anchors import AnchorBox
from efficientdet.data_pipeline.utils import xywh_to_xyxy


class DecodePredictions(tf.keras.layers.Layer):
    """This layer is appended to the detection model after it has
    been trained. This layer decodes the predictions by extracting
    the meaningful bounding boxes.

    Args:
        num_classes (int):
        confidence_threshold (float):
        nms_iou_threshold (float):
        max_detections_per_class (int):
        max_detections (int):
    """

    def __init__(self,
                 num_classes=10,
                 confidence_threshold=0.05,
                 nms_iou_threshold=0.5,
                 max_detections_per_class=100,
                 max_detections=100,
                 **kwargs):

        super(DecodePredictions, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detections_per_class = max_detections_per_class
        self.max_detections = max_detections

        self.anchor_box = AnchorBox()
        self._box_variance = BOX_VARIANCE

    def _decode_box_predictions(self, anchor_boxes, box_predictions):
        """This method adjusts the anchors according to the regression predictions.
        The regression predictions are encoded as:
        [
            x_center_pred = (x_center_gt - x_center_anchor) / width_anchor,
            y_center_pred = (y_center_gt - y_center_anchor) / height_anchor,
            width_pred = log(gt_width / width_anchor),
            height_pred = log(gt_height / height_anchor
        ]

        These prediction are the decoded to obtain x_center_gt (which is still a
        prediction):
        x_center_gt = x_center_pred * width_anchor + x_center_anchor
        y_center_gt = y_center_pred * height_anchor + y_center_anchor
        gt_width = exp(width_pred) * width_anchor
        gt_height = exp(height_pred) * height_anchor

        Finally, these [x_center, y_center, width, height] coordinates are
        transformed to [x_min, y_min, x_max, y_max] coordinates.

        Args:
            anchor_boxes (tf.Tensor):
            box_predictions (tf.Tensor):

        Returns:
            tf.Tensor
        """
        boxes = box_predictions * self._box_variance
        boxes = tf.concat(
            [boxes[:, :, :2] * anchor_boxes[:, :, 2:] + anchor_boxes[:, :, :2],
             tf.math.exp(boxes[:, :, 2:]) * anchor_boxes[:, :, 2:]],
            axis=-1,
        )
        boxes_transformed = xywh_to_xyxy(boxes)
        return boxes_transformed

    def call(self, inputs, **kwargs):
        """Decodes the box predictions and applies non-max suppression to
        keep the meaningful boxes.

        Args:
            inputs (tf.Tensor):
            **kwargs:

        Returns:
            tuple[tf.Tensor, tf.Tensor, tf.Tensor]
        """
        images, box_pred, cls_pred = inputs
        image_shape = tf.cast(tf.shape(images), dtype=tf.float32)
        anchor_boxes = self.anchor_box.get_all_anchors(image_shape[1], image_shape[2])
        boxes = self._decode_box_predictions(anchor_boxes[None, ...], box_pred)

        boxes, scores, classes, _ = tf.image.combined_non_max_suppression(
            tf.expand_dims(boxes, axis=2),
            cls_pred,
            self.max_detections_per_class,
            self.max_detections,
            self.nms_iou_threshold,
            self.confidence_threshold,
            clip_boxes=False
        )

        return boxes, scores, classes
