import logging
import numpy as np
import tensorflow as tf

from image_captioning.data_pipeline import utils
from image_captioning.constants import ANNOTATION_FILE


def load_annotations():
    """Loads annotations from disk and prepares image path with
    caption combinations.

    Returns:
        tuple[list, list, dict]
    """
    logging.info("Loading annotations file from disk.")
    annotations = utils.load_json_file(ANNOTATION_FILE)
    imgpath_to_caption, image_paths = utils.group_captions(annotations)

    all_captions = []
    all_imgpaths = []

    for image_path in image_paths:
        caption_list = imgpath_to_caption[image_path]
        all_captions.extend(caption_list)
        all_imgpaths.extend([image_path] * len(caption_list))  # duplicate image path so that every caption has its own image path

    return all_captions, all_imgpaths, imgpath_to_caption


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


def load_features_from_numpy(features_path, cap):
    """Loads features from disk

    Args:
        features_path (str):
        cap (str):

    Returns:
        tuple[tf.Tensor, str]
    """
    img_tensor = np.load(features_path.decode('utf-8'), allow_pickle=True)
    return img_tensor, cap


def create_input_dataset(feature_paths, captions, tokenizer, buffer_size, batch_size):
    """Creates a tensorflow dataset.

    Args:
        feature_paths (list):
        captions (list):
        tokenizer (tf.keras.layers.TextVectorization):
        buffer_size (int):
        batch_size (int):

    Returns:
        tf.data.Dataset
    """
    dataset = tf.data.Dataset.from_tensor_slices((feature_paths, captions))

    # preprocess the captions
    dataset = dataset.map(lambda path, text: (path, tokenizer(text)))
    # use map to load the numpy files in parallel
    dataset = dataset.map(
        lambda path, text: tf.numpy_function(load_features_from_numpy, [path, text], [tf.float32, tf.int64]),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Shuffle and batch
    dataset = dataset \
        .shuffle(buffer_size) \
        .batch(batch_size) \
        .prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset


def split_dataset(all_imgpaths, imgpath_to_caption, seed=42):
    """Splits the data randomly. A seed is added for reproducibility.

    Args:
        all_imgpaths (list):
        imgpath_to_caption (dict):
        seed (int):

    Returns:
        tuple[list, list, list, list]
    """
    np.random.seed(seed)

    unique_imgpaths = list(set(all_imgpaths))
    np.random.shuffle(unique_imgpaths)

    slice_index = int(len(unique_imgpaths) * 0.8)

    train_featurepaths = []
    train_captions = []
    for imgt in unique_imgpaths[:slice_index]:
        feature_path = utils.imgpath_to_featurepath(imgt)

        capt_len = len(imgpath_to_caption[imgt])
        train_featurepaths.extend([feature_path] * capt_len)
        train_captions.extend(imgpath_to_caption[imgt])

    val_featurepaths = []
    val_captions = []
    for imgv in unique_imgpaths[slice_index:]:
        feature_path = utils.imgpath_to_featurepath(imgv)

        capv_len = len(imgpath_to_caption[imgv])
        val_featurepaths.extend([feature_path] * capv_len)
        val_captions.extend(imgpath_to_caption[imgv])

    return train_featurepaths, train_captions, val_featurepaths, val_captions
