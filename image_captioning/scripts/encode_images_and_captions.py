import os
import sys
import logging
import argparse
import numpy as np
import tensorflow as tf
from glob import glob
from tqdm import tqdm

from image_captioning.data_pipeline import input_dataset, utils
from image_captioning.model import text_vectorization, encoder
from image_captioning.constants import *

logformat = '%(asctime)s | %(levelname)s | [%(filename)s:%(lineno)s - %(funcName)s] %(message)s'
logging.basicConfig(format=logformat, level=logging.INFO, stream=sys.stdout)


def encode_images_to_features(all_imgpaths):
    """Encodes all images with an InceptionV3 network that is
    initialized with imagenet weights.

    Args:
        all_imgpaths (list):
    """
    logging.info("Initialize inceptionV3 network with imagenet weights.")
    inceptionV3 = encoder.create_inception_v3()

    os.makedirs(FEATURES_DIR, exist_ok=True)  # directory where to store processed features

    # get unique image paths that are not processed yet
    img_paths_processed = [p.replace('_features', '').replace('.npy', '.jpg') for p in glob(os.path.join(FEATURES_DIR, '*'))]
    encode_images_list = sorted(set(all_imgpaths) - set(img_paths_processed))
    logging.info(f"Number of images left to process: {len(encode_images_list)}")

    if len(encode_images_list) > 0:
        # create dataset that returns images and their corresponding filepaths
        image_dataset = tf.data.Dataset.from_tensor_slices(encode_images_list)
        image_dataset = image_dataset.map(input_dataset.load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        image_dataset = image_dataset.batch(32)

        for batch_imgs, batch_paths in tqdm(image_dataset):

            batch_features = inceptionV3(batch_imgs)  # output shape (bs, 8, 8, 2048)
            batch_features = tf.reshape(batch_features, (batch_features.shape[0], -1, batch_features.shape[3]))  # output shape (bs, 64, 2048); basically flattens the spatial dimension

            for bf, p in zip(batch_features, batch_paths):
                img_path = p.numpy().decode("utf-8")
                feature_path = utils.imgpath_to_featurepath(img_path)
                np.save(feature_path, bf.numpy())
    logging.info("Finished encoding images!")


def fit_text_vectorizer(all_captions, max_length, vocabulary_size):
    """Fits a TextVectorization tensorflow layer.

    Args:
        all_captions (list):
        max_length (int):
        vocabulary_size (int):
    """
    if os.path.exists(TOKENIZER_PATH):
        logging.info("Tokenizer already exists.")
    else:
        logging.info("Fitting new tokenizer.")
        tokenizer = text_vectorization.fit_text_vectorizer(all_captions, text_vectorization.standardize_text, max_length, vocabulary_size)
        text_vectorization.save_text_vectorizer(tokenizer, TOKENIZER_PATH)
    logging.info("Finished fitting tokenizer!")


def main(max_length, vocabulary_size):
    """Combines all functionality:
    1. loads the necessary data
    2. encodes images to features
    3. fits a TextVectorization layer

    Args:
        max_length (int):
        vocabulary_size (int):
    """
    all_captions, all_imgpaths, imgpath_to_caption = input_dataset.load_annotations()
    encode_images_to_features(all_imgpaths)
    fit_text_vectorizer(all_captions, max_length, vocabulary_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config',
                        default='../config.json',
                        help='Path to configuration file')
    known_args, _ = parser.parse_known_args()
    config = utils.load_json_file(known_args.config)
    main(config['max_text_length'], config['vocabulary_size'])
