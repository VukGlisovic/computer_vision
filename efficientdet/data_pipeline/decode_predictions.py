import tensorflow as tf

from efficientdet.constants import BOX_VARIANCE
from efficientdet.data_pipeline.anchors import AnchorBox
from efficientdet.data_pipeline.utils import xywh_to_xyxy


class DecodePredictions(tf.keras.layers.Layer):

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
        boxes = box_predictions * self._box_variance
        boxes = tf.concat(
            [boxes[:, :, :2] * anchor_boxes[:, :, 2:] + anchor_boxes[:, :, :2],
             tf.math.exp(boxes[:, :, 2:]) * anchor_boxes[:, :, 2:]],
            axis=-1,
        )
        boxes_transformed = xywh_to_xyxy(boxes)
        return boxes_transformed

    def call(self, inputs, **kwargs):
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
                                            clip_boxes=False)

        return boxes, scores, classes
