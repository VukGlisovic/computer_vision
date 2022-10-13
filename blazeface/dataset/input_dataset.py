import os
import tensorflow as tf
import tensorflow_datasets as tfds

from blazeface.dataset import anchors
from blazeface.dataset import target_encoder
from blazeface.constants import *


def load_the300w_lp(split="train[:80%]"):
    """Downloads and prepares the 300w_lp dataset. Running this
    method the first time can take some time.

    Args:
        split (str): indicating what split to take.

    Returns:
        tuple[tf.data.Dataset, tfds.core.DatasetInfo]
    """
    dataset, info = tfds.load("the300w_lp", split=split, data_dir=os.path.join(DATA_DIR, "tensorflow_datasets"), with_info=True)
    return dataset, info


def unpack_dct(dct):
    """Extracts the necessary information from the input dictionary and
    adds another dimension to the landmarks.
    The dictionary provides information per sample.

    Args:
        dct (dict):

    Returns:
        tuple[tf.Tensor, tf.Tensor]
    """
    return dct['image'], tf.expand_dims(dct["landmarks_2d"], 0)


def preprocess_image(img, landmarks):
    """Resizes the image and simply passes along the landmarks.

    Args:
        img (tf.Tensor):
        landmarks (tf.Tensor):

    Returns:
        tuple[tf.Tensor, tf.Tensor]
    """
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE), preserve_aspect_ratio=True)
    return img, landmarks


def reduce_landmarks(landmarks):
    """Reduces the 68 landmarks of the original 300w_lp dataset to
    just 6 landmarks by averaging over the various landmarks.
    Note that the landmarks are relative coordinates (between 0 and 1).

    Args:
        landmarks (tf.Tensor): shape [bs, 68, 2]

    Returns:
        tf.Tensor: shape [bs, 6, 2]
    """
    left_eye = tf.reduce_mean(landmarks[..., 36:42, :], axis=-2)
    right_eye = tf.reduce_mean(landmarks[..., 42:48, :], axis=-2)
    left_ear = tf.reduce_mean(landmarks[..., 0:2, :], axis=-2)
    right_ear = tf.reduce_mean(landmarks[..., 15:17, :], axis=-2)
    nose = tf.reduce_mean(landmarks[..., 27:36, :], axis=-2)
    mouth = tf.reduce_mean(landmarks[..., 48:68, :], axis=-2)
    return tf.stack([left_eye, right_eye, left_ear, right_ear, nose, mouth], axis=-2)


def landmarks_to_bboxes(landmarks):
    """Converts landmarks to a face bounding box. The bounding boxes
    will have values in the [0, 1] range.

    Args:
        landmarks (tf.Tensor): shape [bs, 68, 2]

    Returns:
        tf.Tensor: shape [bs, 4]
    """
    x1 = tf.reduce_min(landmarks[..., 0], axis=-1) - BBOX_PADDING
    y1 = tf.reduce_min(landmarks[..., 1], axis=-1) - BBOX_PADDING
    x2 = tf.reduce_max(landmarks[..., 0], axis=-1) + BBOX_PADDING
    y2 = tf.reduce_max(landmarks[..., 1], axis=-1) + BBOX_PADDING
    bboxes = tf.stack([x1, y1, x2, y2], axis=-1)
    return tf.clip_by_value(bboxes, 0, 1)


def create_input_dataset(dataset, batch_size=12):
    """Applies the necessary preprocessing steps to run a blazeface training.

    Args:
        dataset (tf.data.Dataset):
        batch_size (int):

    Returns:
        tf.data.Dataset
    """
    ds = dataset.map(unpack_dct)
    ds = ds.map(preprocess_image)
    ds = ds.map(lambda img, lmarks: (img, landmarks_to_bboxes(lmarks), reduce_landmarks(lmarks)))
    ds = ds.batch(batch_size)
    all_anchors = anchors.generate_anchors()
    ds = ds.map(lambda img, bboxes, lmarks: (img, target_encoder.calculate_targets(all_anchors, bboxes, lmarks)))
    return ds
