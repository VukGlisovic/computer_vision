import os
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
import tensorflow as tf


def load_img(path_to_img, max_dim=99999):
    """Loads an image from path_to_img. The image is directly
    scaled to the [0,1] interval. If necessary decreases the
    shape of the image while preserving the aspect ratio.

    Args:
        path_to_img (str):
        max_dim (int): if the size of the largest side is greater
            than max_dim, then the image gets decreased in size such
            that the largest side has size max_dim. Aspect ratios
            are preserved.

    Returns:
        tf.Tensor
    """
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)  # this operation also scales the pixel values

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    if long_dim > max_dim:
        scale = max_dim / long_dim
        new_shape = tf.cast(shape * scale, tf.int32)
        img = tf.image.resize(img, new_shape)

    img = img[tf.newaxis, :]
    return img


def tensor_to_image(tensor):
    """Converts a tensor to a PIL image. It expects that
    the image tensor is not scaled to its original value
    range yet.

    Args:
        tensor (tf.Tensor):

    Returns:
        PIL.Image.Image
    """
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


def imshow(image, title=None):
    """Plots an image.

    Args:
        image (tf.Tensor):
        title (str):
    """
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)  # convert shape to (height, width, channels)

    plt.imshow(image)
    if title:
        plt.title(title)


def store_tensor_image(tensor, path):
    """Stores an image tensor in path. If the directory
    doesn't exist yet, then it will be created.

    Args:
        tensor (tf.Tensor):
        path (str):
    """
    _dir = path.rsplit('/', 1)[0]
    os.makedirs(_dir, exist_ok=True)
    pil_img = tensor_to_image(tensor)
    pil_img.save(path)
