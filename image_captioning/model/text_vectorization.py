import os
import pickle
import tensorflow as tf


def standardize_text(text):
    """In order to preserve the '<>' tokens for <start> and <end>,
    we will override the default standardization of TextVectorization.

    Args:
        text (Union[str, tf.Tensor]):

    Returns:
        tf.Tensor
    """
    text = tf.strings.lower(text)
    return tf.strings.regex_replace(text, r"!\"#$%&\(\)\*\+.,-/:;=?@\[\\\]^_`{|}~", "")


def fit_text_vectorizer(texts, standardize, max_length, vocabulary_size):
    """Fits a TextVectorization layer

    Args:
        texts (list[str]):
        standardize (Callable):
        max_length (int):
        vocabulary_size (int):

    Returns:
        tf.keras.layers.TextVectorization
    """
    text_dataset = tf.data.Dataset.from_tensor_slices(texts)  # create tf dataset

    tokenizer = tf.keras.layers.TextVectorization(
        max_tokens=vocabulary_size,
        standardize=standardize,
        output_sequence_length=max_length
    )

    tokenizer.adapt(text_dataset)  # Learn the vocabulary from the caption data
    return tokenizer


def save_text_vectorizer(text_vectorizer, path):
    """Saving TextVectorization object to pickle file.

    Code base on:
    https://stackoverflow.com/questions/65103526/how-to-save-textvectorization-to-disk-in-tensorflow

    Args:
        text_vectorizer (tf.keras.layers.TextVectorization):
        path (str):
    """
    # Pickle the config and weights
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pickle.dump(
        {'config': text_vectorizer.get_config(), 'weights': text_vectorizer.get_weights()},
        open(path, "wb")
    )


def load_text_vectorizer(path):
    """Creates a TextVectorization object and loads the fitted data
    from disk into the newly created object.

    Code based on:
    https://stackoverflow.com/questions/65103526/how-to-save-textvectorization-to-disk-in-tensorflow

    Args:
        path (str):

    Returns:
        tf.keras.layers.TextVectorization
    """
    from_disk = pickle.load(open(path, "rb"))
    text_vectorizer = tf.keras.layers.TextVectorization.from_config(from_disk['config'])
    # You have to call `adapt` with some dummy data (BUG in Keras)
    text_vectorizer.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
    text_vectorizer.set_weights(from_disk['weights'])
    return text_vectorizer
