from unet.model.constants import *
from glob import glob
import numpy as np
import cv2
import tensorflow as tf


def load_data(image_path_glob='../data/images/*', mask_path_glob='../data/masks/*'):
    # get all file paths
    train_image_files = glob(image_path_glob)
    train_mask_files = glob(mask_path_glob)
    # load the image data
    Xtrain = np.array([cv2.imread(p)[:, :, :1] for p in train_image_files])  # only one of the three channels is needed
    ytrain = np.array([cv2.imread(p)[:, :, :1] for p in train_mask_files])
    return Xtrain, ytrain


def preprocess_image(image, mask):
    image = tf.dtypes.cast(image, dtype=tf.dtypes.float32) / 255.
    mask = tf.dtypes.cast(mask, dtype=tf.dtypes.float32) / 255.
    return image, mask


def input_fn(Xtrain, ytrain, epochs=None, batch_size=32, shuffle_buffer=None):
    dataset = tf.data.Dataset.from_tensor_slices((Xtrain, ytrain))
    dataset = dataset.map(preprocess_image)
    dataset = dataset.repeat(epochs)
    if shuffle_buffer:
        dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch_size)
    return dataset
