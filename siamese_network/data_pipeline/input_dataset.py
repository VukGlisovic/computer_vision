import logging
import numpy as np
import tensorflow as tf
from siamese_network.constants import IMG_SPATIAL_SIZE


def load_image(path):
    """Loads an image from disk.

    Args:
        path (str):

    Returns:
        tf.Tensor
    """
    image_string = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image_string, channels=3)
    return image


def load_and_preprocess_image(path):
    """Loads and preprocesses an image from disk.

    Args:
        path (str):

    Returns:
        tf.Tensor
    """
    image = load_image(path)
    image = tf.image.convert_image_dtype(image, tf.float32)  # this scales te images to the [0,1] range
    image = tf.image.resize(image, IMG_SPATIAL_SIZE)
    # image = resnet.preprocess_input(image)
    return image


def preprocess_triplet(a, p, n):
    """Loads and preprocesses a triplet of images from disk.

    Args:
        a (str): path to anchor image
        p (str): path to positive image
        n (str): path to negative image

    Returns:
        tuple[tf.Tensor, tf.Tensor, tf.Tensor]
    """
    return load_and_preprocess_image(a), load_and_preprocess_image(p), load_and_preprocess_image(n)


def create_images_dataset(path_list, shuffle_list=False, map_preprocessing_fnc=False):
    """Creates a tensorflow dataset from a list of image paths.
    Optionally shuffle the list before creating the tf dataset
    and optionally loads and preprocesses the image as well.

    Args:
        path_list (list[str]):
        shuffle_list (bool):
        map_preprocessing_fnc (bool):

    Returns:
        tf.data.Dataset
    """
    if shuffle_list:
        np.random.RandomState(seed=42).shuffle(path_list)
    ds = tf.data.Dataset.from_tensor_slices(path_list)
    if map_preprocessing_fnc:
        ds = ds.map(load_and_preprocess_image)
    return ds


def create_triplet_dataset(anchor_images, positive_images, batch_size=32):
    """Creates a training and validation tensorflow dataset for training
    a siamese network.

    Args:
        anchor_images (list[str]):
        positive_images (list[str]):
        batch_size (int):

    Returns:
        tuple[tf.data.Dataset, tf.data.Dataset]
    """
    # create anchor and positive dataset
    ds_anchor = create_images_dataset(anchor_images)
    ds_positive = create_images_dataset(positive_images)

    # use both anchor and positive images, shuffle the images so that you get a list of negative images
    negative_images = anchor_images + positive_images
    logging.info("Number of negative images: %s", len(negative_images))
    ds_negative = create_images_dataset(negative_images, shuffle_list=True)
    ds_negative = ds_negative.shuffle(buffer_size=4096)

    # combine anchor, positive and negative dataset into one full dataset
    ds_full = tf.data.Dataset.zip((ds_anchor, ds_positive, ds_negative))
    ds_full = ds_full.shuffle(buffer_size=1024)
    ds_full = ds_full.map(preprocess_triplet)

    # split the dataset into train and validation
    split_point = round(len(anchor_images) * 0.8)
    ds_train = ds_full.take(split_point)
    ds_validation = ds_full.skip(split_point)

    # apply batching and prefetch for speed
    ds_train = ds_train.batch(batch_size, drop_remainder=False)
    ds_train = ds_train.prefetch(8)

    ds_validation = ds_validation.batch(batch_size, drop_remainder=False)
    ds_validation = ds_validation.prefetch(8)

    return ds_train, ds_validation
