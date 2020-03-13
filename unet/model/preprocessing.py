import logging
from glob import glob

import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split

from unet.model.augmentations import *


def load_data(image_path_glob='../data/train/images/*', mask_path_glob='../data/train/masks/*'):
    """Loads the data from disk. It returns the loaded images with their
    corresponding masks for segmentation. Note that the output size of
    Xtrain and ytrain will be [nr_samples, 101, 101, 1].

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
    """Preprocesses an image by scaling the pixels to the [0, 1] range.

    Args:
        image (tf.Tensor):
        mask (tf.Tensor):

    Returns:
        tuple[tf.Tensor, tf.Tensor]
    """
    image = tf.dtypes.cast(image, dtype=tf.dtypes.float32) / 255.
    mask = tf.dtypes.cast(mask, dtype=tf.dtypes.float32) / 255.
    return image, mask


def input_fn(Xtrain, ytrain, epochs=None, batch_size=32, shuffle_buffer=None, augment=False):
    """Creates a tensorflow data set iterator as a preprocessing input pipeline.
    Steps:
    1. creates a tf.data.Dataset
    2. maps images to the preprocessing function
    3. sets the number of epochs, batch size and optionally the shuffle buffer

    Args:
        Xtrain (np.ndarray):
        ytrain (np.ndarray):
        epochs (int):
        batch_size (int):
        shuffle_buffer (int):
        augment (bool):

    Returns:
        tf.data.Dataset
    """
    dataset = tf.data.Dataset.from_tensor_slices((Xtrain, ytrain))
    dataset = dataset.map(preprocess_image)
    if augment:
        dataset = dataset.map(rotate)
        dataset = dataset.map(flip)
    dataset = dataset.repeat(epochs)
    if shuffle_buffer:
        dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch_size)
    return dataset


def create_data_generators(nr_epochs, batch_size, shuffle_buffer):
    """This method creates two data generators; one for training and one for
    validation (during training). It already fixes the number of training
    examples to be 3200 and therefore the number of testing examples to be
    800.

    Args:
        nr_epochs (int):
        batch_size (int):
        shuffle_buffer (int):

    Returns:
        tuple[tf.data.Dataset, tf.data.Dataset]
    """
    logging.info("Loading the data.")
    Xdata, ydata = load_data()  # expecting 4000 samples in a numpy array
    logging.info("Splitting the data into train and validation set.")
    train_size = 3200
    Xtrain, Xvalid, ytrain, yvalid = train_test_split(Xdata, ydata, train_size=train_size, random_state=42)
    logging.info("Creating train data input_fn.")
    train_dataset = input_fn(Xtrain, ytrain, epochs=nr_epochs, batch_size=batch_size, shuffle_buffer=shuffle_buffer, augment=True)
    logging.info("Creating validation data input_fn.")
    valid_dataset = input_fn(Xvalid, yvalid, epochs=None, batch_size=200, shuffle_buffer=None, augment=False)
    return train_dataset, valid_dataset
