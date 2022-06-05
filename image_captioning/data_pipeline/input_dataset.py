import tensorflow as tf


def load_image(path):
    """Loads an image from disk.

    Args:
        path (str):

    Returns:
        tf.Tensor
    """
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3)
    return img


def preprocess_image(img):
    """Resizes the image to the spatial size that InceptionV3 expects
    and adjust the dynamic range to [-1, 1].

    Args:
        img (tf.Tensor):

    Returns:
        tf.Tensor
    """
    img = tf.image.resize(img, [299, 299])
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img


def load_and_preprocess_image(path):
    """Loads and preprocesses an image from disk.

    Args:
        path (str):

    Returns:
        tf.Tensor
    """
    img = load_image(path)
    img = preprocess_image(img)
    return img, path
