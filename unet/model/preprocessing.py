from unet.model.constants import *
from glob import glob
import numpy as np
import cv2
import tensorflow as tf


def load_data(image_path_glob='../data/images/*', mask_path_glob='../data/masks/*'):
    """Loads the image and mask data from local disk.

    Args:
        image_path_glob (str):
        mask_path_glob (str):

    Returns:
        tuple[np.ndarray, np.ndarray]
    """
    # get all file paths
    train_image_files = glob(image_path_glob)
    train_mask_files = glob(mask_path_glob)
    # load the image data
    Xtrain = np.array([cv2.imread(p)[:, :, :1] for p in train_image_files])  # only one of the three channels is needed
    ytrain = np.array([cv2.imread(p)[:, :, :1] for p in train_mask_files])
    return Xtrain, ytrain


def preprocess_image(image, mask):
    """Applies preprocessing steps to the image and mask.

    Args:
        image (tf.Tensor):
        mask (tf.Tensor):

    Returns:
        tuple[tf.Tensor, tf.Tensor]
    """
    image = tf.dtypes.cast(image, dtype=tf.dtypes.float32) / 255.
    mask = tf.dtypes.cast(mask, dtype=tf.dtypes.float32) / 255.
    return image, mask


def input_fn(Xtrain, ytrain, epochs=None, batch_size=32, shuffle_buffer=None):
    """Creates an input tensorflow dataset iterator.

    Args:
        Xtrain (np.ndarray):
        ytrain (np.ndarray):
        epochs (int):
        batch_size (int):
        shuffle_buffer (int):

    Returns:
        tf.dataset.Dataset
    """
    dataset = tf.data.Dataset.from_tensor_slices((Xtrain, ytrain))
    dataset = dataset.map(preprocess_image)
    dataset = dataset.repeat(epochs)
    if shuffle_buffer:
        dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch_size)
    return dataset
