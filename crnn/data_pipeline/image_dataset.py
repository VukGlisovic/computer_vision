import logging
from crnn.data_pipeline.augmentations import *
from crnn.constants import *


def create_images_dataset(df, augment):
    ds_images = tf.data.Dataset.from_tensor_slices(df[FILENAME].values)
    # apply preprocessing
    ds_images = ds_images.map(load_and_preprocess_image)
    if augment:
        logging.info("Adding augmentations.")
        ds_images = apply_augmentations(ds_images)
    ds_images = ds_images.map(pad_image)
    return ds_images


def load_image(filename):
    img = tf.io.read_file(tf.strings.join([DIR_IMAGES, filename]))
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3, fancy_upscaling=False, dct_method="INTEGER_ACCURATE")
    return img


def preprocess_image(img):
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    img = tf.image.resize(img, (IM_HEIGHT, IM_WIDTH), preserve_aspect_ratio=True)
    # transpose width and height
    img = tf.image.transpose(img)
    img = tf.image.flip_left_right(img)
    return img


def load_and_preprocess_image(path):
    img = load_image(path)
    img = preprocess_image(img)
    return img


def pad_image(img):
    # width and height are flipped because the image was flipped if all is well
    paddings = [[0, IM_WIDTH - tf.shape(img)[0]], [0, IM_HEIGHT - tf.shape(img)[1]], [0, 0]]
    return tf.pad(img, paddings, "CONSTANT", constant_values=1)
