import os
import sys
import logging
import argparse
import numpy as np
import tensorflow as tf
from glob import glob
from tqdm import tqdm

from image_captioning.data_pipeline import input_dataset
from image_captioning.data_pipeline import utils
from image_captioning.model import text_vectorization
from image_captioning.constants import *

logformat = '%(asctime)s | %(levelname)s | [%(filename)s:%(lineno)s - %(funcName)s] %(message)s'
logging.basicConfig(format=logformat, level=logging.INFO, stream=sys.stdout)


def load_annotations():
    logging.info("Loading annotations file from disk.")
    annotations = utils.load_json_file(ANNOTATION_FILE)
    imgpath_to_caption, image_paths = utils.group_captions(annotations)

    all_captions = []
    all_imgpaths = []

    for image_path in image_paths:
        caption_list = imgpath_to_caption[image_path]
        all_captions.extend(caption_list)
        all_imgpaths.extend([image_path] * len(caption_list))  # duplicate image path so that every caption has its own image path

    return all_captions, all_imgpaths


def encode_images_to_features(all_imgpaths):
    logging.info("Initialize inceptionV3 network with imagenet weights.")
    inceptionV3 = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')

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
    tokenizer_path = os.path.join(DATA_DIR, 'experiment/tokenizer.pkl')
    if os.path.exists(tokenizer_path):
        logging.info("Tokenizer already exists.")
        tokenizer = text_vectorization.load_text_vectorizer(tokenizer_path)
    else:
        logging.info("Fitting new tokenizer.")
        tokenizer = text_vectorization.fit_text_vectorizer(all_captions, text_vectorization.standardize_text, max_length, vocabulary_size)
        text_vectorization.save_text_vectorizer(tokenizer, tokenizer_path)
    logging.info("Finished fitting tokenizer!")


def main(max_length, vocabulary_size):
    all_captions, all_imgpaths = load_annotations()
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
