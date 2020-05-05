import os
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
import tensorflow as tf


def load_img(path_to_img):
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img[tf.newaxis, :]
    return img


def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


def imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)  # convert shape to (height, width, channels)

    plt.imshow(image)
    if title:
        plt.title(title)


def store_image(image, path):
    _dir = path.rsplit('/', 1)[0]
    os.makedirs(_dir, exist_ok=True)
    pil_img = tensor_to_image(image)
    pil_img.save(path)
