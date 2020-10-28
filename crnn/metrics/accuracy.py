import tensorflow as tf
from tensorflow import keras
from crnn.constants import *


class Accuracy(keras.metrics.Metric):

    def __init__(self, method='greedy', name='accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.method = method
        self.total = self.add_weight(name='total', dtype=tf.int32, initializer=tf.zeros_initializer())
        self.total_correct = self.add_weight(name='count', dtype=tf.int32, initializer=tf.zeros_initializer())

    def update_state(self, y_true, y_pred, sample_weight=None):
        batch_count = tf.shape(y_true)[0]
        max_width = tf.maximum(tf.shape(y_true)[1], tf.shape(y_pred)[1])
        logit_length = tf.fill([tf.shape(y_pred)[0]], tf.shape(y_pred)[1])
        if self.method == 'greedy':
            decoded, _ = tf.nn.ctc_greedy_decoder(inputs=tf.transpose(y_pred, perm=[1, 0, 2]), sequence_length=logit_length)
        elif self.method == 'beam_search':
            decoded, _ = tf.nn.ctc_beam_search_decoder(inputs=tf.transpose(y_pred, perm=[1, 0, 2]), sequence_length=logit_length)
        else:
            raise ValueError("Wrong decoding method configured.")
        # prepare labels
        idx = tf.where(tf.not_equal(y_true, BLANK_INDEX))  # Find indices where the tensor is not blank character
        y_true = tf.SparseTensor(idx, tf.gather_nd(y_true, idx), tf.shape(y_true, out_type=tf.int64))  # create sparse tensor
        y_true = tf.sparse.reset_shape(y_true, [batch_count, max_width])
        y_true = tf.sparse.to_dense(y_true, default_value=BLANK_INDEX)
        y_true = tf.cast(y_true, tf.int32)
        # prepare predictions
        y_pred = tf.sparse.reset_shape(decoded[0], [batch_count, max_width])
        y_pred = tf.sparse.to_dense(y_pred, default_value=BLANK_INDEX)
        y_pred = tf.cast(y_pred, tf.int32)
        # calculate number of wrong predictions
        wrong_predictions = tf.math.reduce_any(tf.math.not_equal(y_true, y_pred), axis=1)
        wrong_predictions = tf.cast(wrong_predictions, tf.int32)
        nr_wrong_predictions = tf.reduce_sum(wrong_predictions)
        self.total.assign_add(batch_count)
        self.total_correct.assign_add(batch_count - nr_wrong_predictions)

    def result(self):
        return self.total_correct / self.total

    def reset_states(self):
        self.total_correct.assign(0)
        self.total.assign(0)
