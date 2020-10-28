import tensorflow as tf
from crnn.data_pipeline.decoder import Decoder
from crnn.model.ctc_loss import label_lengths
from crnn.constants import MAX_CHARACTERS


def to_sparse(tensor, lengths):
    mask = tf.sequence_mask(lengths, MAX_CHARACTERS)
    indices = tf.cast(tf.where(tf.equal(mask, True)), tf.int64)
    values = tf.cast(tf.boolean_mask(tensor, mask), tf.int32)
    shape = tf.cast(tf.shape(tensor), tf.int64)
    return tf.SparseTensor(indices, values, shape)


def get_normalized_edit_distance():

    decoder = Decoder()

    def normalized_edit_distance(y_true, y_pred):
        # prepare predictions
        y_pred_sparse = decoder.decode(y_pred, from_pred=True, method='beam_search', to_text=False)
        # prepare ground truth
        label_length = label_lengths(y_true)
        y_true_sparse = to_sparse(y_true, label_length)
        # calculate edit distance
        distances = tf.edit_distance(tf.cast(y_pred_sparse, tf.int32), tf.cast(y_true_sparse, tf.int32), normalize=True)
        return tf.reduce_mean(distances)

    return normalized_edit_distance
