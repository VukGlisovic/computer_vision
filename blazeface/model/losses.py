import tensorflow as tf

from blazeface.constants import NEG_POS_RATIO, LOC_LOSS_ALPHA, N_LANDMARKS


class ClassLoss(tf.keras.losses.Loss):

    def __init__(self):
        super().__init__(name='class_loss')
        self.neg_pos_ratio = tf.constant(NEG_POS_RATIO, dtype=tf.float32)
        self.bxe = tf.losses.BinaryCrossentropy(reduction=tf.losses.Reduction.NONE)

    def call(self, y_true, y_pred):
        """Calculating SSD confidence loss and using hard negative mining.
        Computes the binary cross entropy loss for all positive anchors and
        does hard negative mining for the negative anchors.

        Args:
            y_true (tf.Tensor): shape [bs, n_anchors, 1]
            y_pred (tf.Tensor): shape [bs, n_anchors, 1]

        Returns:
            tf.Tensor: scalar value
        """
        # Confidence / Label loss calculation for all labels
        cross_entropy_losses = self.bxe(y_true, y_pred)
        # find positive anchor boxes
        y_true_squeezed = tf.squeeze(y_true, axis=-1)  # remove final dimension -> shape [bs, n_anchors]
        pos_mask = tf.not_equal(y_true_squeezed, 0.)
        pos_mask = tf.cast(pos_mask, dtype=tf.float32)
        n_pos_bboxes = tf.reduce_sum(pos_mask, axis=1)  # shape [bs]
        # find negative anchor boxes with hard negative mining
        n_neg_bboxes = tf.cast(n_pos_bboxes * self.neg_pos_ratio, tf.int32)
        masked_loss = tf.where(tf.equal(y_true_squeezed, 0.), cross_entropy_losses, tf.zeros_like(cross_entropy_losses, dtype=tf.float32))
        # yields a tensor where the losses are ordered and their indices are taken
        sorted_indices_of_loss = tf.argsort(masked_loss, axis=-1, direction="DESCENDING")
        # yields a tensor where each location has an index indicating what rank the loss value is in in that specific location
        sorted_indices_of_loss = tf.argsort(sorted_indices_of_loss, axis=-1)
        neg_mask = tf.less(sorted_indices_of_loss, tf.expand_dims(n_neg_bboxes, axis=1))
        neg_mask = tf.cast(neg_mask, dtype=tf.float32)
        # merge positive and negative masks and calculate total loss
        full_mask = pos_mask + neg_mask
        class_loss = tf.reduce_sum(full_mask * cross_entropy_losses, axis=-1)
        n_pos_bboxes = tf.where(tf.equal(n_pos_bboxes, 0.), 1., n_pos_bboxes)  # prevent division by zero if no positive bboxes
        class_loss = class_loss / n_pos_bboxes

        return class_loss


class RegressionLoss(tf.keras.losses.Loss):

    def __init__(self):
        super().__init__(name='regression_loss')
        self.loc_loss_alpha = tf.constant(LOC_LOSS_ALPHA, dtype=tf.float32)
        self.total_regression_points = 4 + N_LANDMARKS * 2  # 4 box coordinates and landmark coordinates
        self.huber = tf.losses.Huber(reduction=tf.losses.Reduction.NONE)

    def call(self, y_true, y_pred):
        """Calculating SSD regression loss for positive anchors only.

        Args:
            y_true (tf.Tensor): shape [bs, n_anchors, [delta_center_x, delta_center_y, delta_w, delta_h, delta_lmark_x0, delta_lmark_y0, ..., delta_lmark_xN, delta_lmark_yN]]
            y_pred (tf.Tensor): shape [bs, n_anchors, [delta_center_x, delta_center_y, delta_w, delta_h, delta_lmark_x0, delta_lmark_y0, ..., delta_lmark_xN, delta_lmark_yN])

        Returns:
            tf.Tensor: loc_loss = localization / regression / bounding box loss value
        """
        # Localization / bbox / regression loss calculation for all bboxes
        loc_loss_for_all = self.huber(y_true, y_pred)  # Huber calculates mean over last axis
        loc_loss_for_all = loc_loss_for_all * tf.cast(self.total_regression_points, dtype=tf.float32)
        # find positive anchors
        pos_mask = tf.reduce_any(tf.not_equal(y_true, 0.), axis=2)
        pos_mask = tf.cast(pos_mask, dtype=tf.float32)
        n_pos_bboxes = tf.reduce_sum(pos_mask, axis=1)  # number of positive anchors per sample in batch
        # calculate regression loss over positive anchors only
        loc_loss = tf.reduce_sum(pos_mask * loc_loss_for_all, axis=-1)
        n_pos_bboxes = tf.where(tf.equal(n_pos_bboxes, 0.), 1., n_pos_bboxes)  # prevent division by zero if no positive bboxes
        loc_loss = (loc_loss / n_pos_bboxes) * self.loc_loss_alpha

        return loc_loss
