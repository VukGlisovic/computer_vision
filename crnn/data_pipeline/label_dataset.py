import tensorflow as tf
from crnn.constants import *


def create_label_dataset(df):
    characters_to_label = tf.lookup.TextFileInitializer(
        CLASSES_PATH,
        key_dtype=tf.string,
        key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
        value_dtype=tf.int64,
        value_index=tf.lookup.TextFileIndex.LINE_NUMBER,
        vocab_size=None
    )
    table = tf.lookup.StaticHashTable(
        characters_to_label,
        default_value=-1
    )

    def text_to_label(string):
        """Splits string into individual characters.

        Args:
            string (tf.Tensor):

        Returns:
            tf.Tensor
        """
        characters = tf.strings.unicode_split(string, 'UTF-8')
        return tf.cast(tf.ragged.map_flat_values(table.lookup, characters), tf.float32)

    ds_labels = tf.data.Dataset.from_tensor_slices(df[LABEL].values)
    ds_labels = ds_labels.map(text_to_label)
    ds_labels = ds_labels.map(pad_label)
    return ds_labels


def pad_label(label):

    def pad(lab):
        paddings = [[0, MAX_CHARACTERS - tf.shape(lab)[0]]]
        return tf.pad(lab, paddings, "CONSTANT", constant_values=BLANK_INDEX)

    def truncate(lab):
        return tf.slice(lab, [0], [MAX_CHARACTERS])

    return tf.cond(tf.size(label) < MAX_CHARACTERS, lambda: pad(label), lambda: truncate(label))
