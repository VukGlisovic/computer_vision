from unet.model.constants import *
from glob import glob
import cv2
import tensorflow as tf


def load_data(image_path_glob='../data/images/*', mask_path_glob='../data/masks/*'):
    # get all file paths
    train_image_files = glob(image_path_glob)
    train_mask_files = glob(mask_path_glob)
    # load the image data
    Xtrain = [cv2.imread(p) for p in train_image_files]
    ytrain = [cv2.imread(p) for p in train_mask_files]
    return Xtrain, ytrain


def preprocess_image(image):
    image = image / 255.
    return image


def input_fn(Xtrain, ytrain, epochs=None, batch_size=32, shuffle_buffer=None):
    dataset = tf.data.Dataset.from_tensor_slices((Xtrain, ytrain))
    dataset = dataset.map(preprocess_image) \
        .repeat(epochs) \
        .shuffle(shuffle_buffer) \
        .batch(batch_size)
    return dataset
