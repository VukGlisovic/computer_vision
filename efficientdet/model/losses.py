import tensorflow as tf
from tensorflow.keras.losses import Huber
from tensorflow_addons.losses import SigmoidFocalCrossEntropy
from tensorflow import keras


class HuberRegressionLoss(tf.losses.Loss):
    """Bounding box regression loss.

    This loss basically calculates the distance between the anchor and
    the true bounding box. I.e. how much do delta_x, delta_y, delta_width,
    delta_height need to change from anchor to match the true bounding box.

    Args:
        delta (float):
    """

    def __init__(self, delta=1.0):
        super().__init__()
        self.huber = Huber(delta=delta, reduction='none')

    def call(self, y_true, y_pred):
        """Calculates the regression loss for positive anchors. A loss of
        0.0 is assigned to all other anchors.

        Args:
            y_true (tf.Tensor):
            y_pred (tf.Tensor):

        Returns:
            tf.Tensor
        """
        positive_mask = y_true[:, :, 4]
        y_true = y_true[:, :, :4]
        box_loss = self.huber(y_true, y_pred)
        box_loss = tf.where(tf.equal(positive_mask, 1.0), box_loss, 0.0)
        normalizer = tf.reduce_sum(positive_mask, axis=-1)
        box_loss = tf.math.divide_no_nan(tf.reduce_sum(box_loss, axis=-1), normalizer)
        return box_loss


class FocalClassificationLoss(tf.losses.Loss):
    """Cross Entropy Focal loss.

    The focal loss is taken from https://arxiv.org/abs/1708.02002 (retinanet paper).

    This loss basically gives more weight to harder to classify samples and
    gives less weight to easier to classify samples. This is very useful for
    detection networks since there usually are many background classes as labels.

    Args:
        num_classes (int):
        alpha (float): class weight balancer.
        gamma (float): the higher the value of gamma, the lower the loss for well-classified
            samples. for gamma = 0, the focal loss is equivalent to categorical cross-entropy.
    """

    def __init__(self, num_classes, alpha=0.25, gamma=2.0):
        super().__init__()
        self.num_classes = num_classes
        self.focal = SigmoidFocalCrossEntropy(alpha=alpha, gamma=gamma, reduction='none')

    def call(self, y_true, y_pred):
        """Calculates the classification loss for positive and negative
        anchors. A loss of 0.0 is assigned to anchors that should be ignored.

        Args:
            y_true (tf.Tensor):
            y_pred (tf.Tensor):

        Returns:
            tf.Tensor
        """
        positive_mask = tf.cast(tf.greater(y_true, -1.0), dtype=tf.float32)
        ignore_mask = tf.cast(tf.equal(y_true, -2.0), dtype=tf.float32)
        y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=self.num_classes, dtype=tf.float32)
        cls_loss = self.focal(y_true, y_pred)
        cls_loss = tf.where(tf.equal(ignore_mask, 1.0), 0.0, cls_loss)
        normalizer = tf.reduce_sum(positive_mask, axis=-1)
        cls_loss = tf.math.divide_no_nan(tf.reduce_sum(cls_loss, axis=-1), normalizer)
        return cls_loss
