import tensorflow as tf
from crnn.constants import *


class Decoder:

    def __init__(self):
        self.char_list = tf.constant([IDX_TO_CHAR[i] for i in sorted(IDX_TO_CHAR.keys())], dtype=tf.string)
        self.blank_character = IDX_TO_CHAR[BLANK_INDEX]

    def tf_map2string(self, inputs):
        inputs = tf.gather(self.char_list, inputs)
        strings = tf.strings.reduce_join(inputs, axis=1)
        strings = tf.strings.regex_replace(strings, pattern=self.blank_character, rewrite='', replace_global=True)
        strings = tf.reshape(strings, shape=(-1,1))
        return strings

    def decode(self, inputs, from_pred=True, method='greedy', to_text=True):
        if from_pred:
            logit_length = tf.fill([tf.shape(inputs)[0]], tf.shape(inputs)[1])
            if method == 'greedy':
                decoded, _ = tf.nn.ctc_greedy_decoder(inputs=tf.transpose(inputs, perm=[1, 0, 2]),
                                                      sequence_length=logit_length)
            elif method == 'beam_search':
                decoded, _ = tf.nn.ctc_beam_search_decoder(inputs=tf.transpose(inputs, perm=[1, 0, 2]),
                                                           sequence_length=logit_length)
            else:
                raise ValueError("Wrong decoding method configured.")
            inputs = decoded[0]
        decoded = inputs
        if to_text:
            decoded = tf.sparse.to_dense(decoded, default_value=BLANK_INDEX)
            decoded = self.tf_map2string(decoded)
        return decoded
