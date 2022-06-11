import tensorflow as tf


cxe = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')


def loss_function(real, pred):
    """Calculates the loss between real word and the prediction.

    Args:
        real (tf.Tensor): the target word
        pred (tf.Tensor): prediction vector with probabilities for each word

    Returns:
        tf.Tensor
    """
    mask = tf.math.logical_not(tf.math.equal(real, 0))

    loss_ = cxe(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)
