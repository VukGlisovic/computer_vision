import tensorflow as tf
from tensorflow import keras
from crnn.constants import BLANK_INDEX


def label_lengths(labels):
    """Calculates the number of non bank index characters.

    Args:
        labels (tf.Tensor):

    Returns:
        tf.Tensor
    """
    return tf.reduce_sum(tf.cast(labels != BLANK_INDEX, tf.int32), axis=1)


class CTCLoss(keras.losses.Loss):
    """The Connectionist Temporal Classification (CTC) loss function.

    Args:
        reduction (str):
        name (str):
    """

    def __init__(self, reduction=keras.losses.Reduction.AUTO, name='ctc_loss'):
        super().__init__(reduction=reduction, name=name)

    def call(self, y_true, y_pred):
        """Returns the average CTC loss. Note that the prediction dimensions
        are switched: [batch_size, time_steps, logits] -> [time_steps, batch_size, logits]

        Args:
            y_true (tf.Tensor):
            y_pred (tf.Tensor):

        Returns:
             tf.Tensor
        """
        y_true = tf.cast(y_true, tf.int32)
        logit_length = tf.fill([tf.shape(y_pred)[0]], tf.shape(y_pred)[1])
        label_length = label_lengths(y_true)
        loss = tf.nn.ctc_loss(
            labels=y_true,
            logits=tf.transpose(y_pred, perm=[1, 0, 2]),
            label_length=label_length,
            logit_length=logit_length,
            logits_time_major=True,
            blank_index=BLANK_INDEX
        )
        return tf.reduce_mean(loss)
