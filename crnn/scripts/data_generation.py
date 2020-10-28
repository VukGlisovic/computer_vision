import os
import sys
import logging
import argparse
import pandas as pd
from uuid import uuid1
from tqdm import tqdm
from trdg.generators import GeneratorFromRandom
from crnn.constants import PROJECT_PATH

logformat = '%(asctime)s | %(levelname)s | [%(filename)s:%(lineno)s - %(funcName)s] %(message)s'
logging.basicConfig(format=logformat, level=logging.INFO, stream=sys.stdout)

parser = argparse.ArgumentParser()
parser.add_argument('--train_size',
                    type=int,
                    default=3000,
                    help="The number of images to generate for training.")
parser.add_argument('--validation_size',
                    type=int,
                    default=250,
                    help="The number of images to generate for validation.")
parser.add_argument('--test_size',
                    type=int,
                    default=250,
                    help="The number of images to generate for testing.")
parser.add_argument('--df_dir',
                    type=str,
                    default='../data/processed',
                    help="Where to store train/validation/test data frames.")
parser.add_argument('--images_dir',
                    type=str,
                    default='../data/images',
                    help="Where to store train/validation/test images.")
known_args, _ = parser.parse_known_args()
logging.info("Given input parameters:\n%s", "\n".join(["{}: {}".format(k, v) for k,v in known_args.__dict__.items()]))


def create_random_image_generator(count):
    """Creates a data generator that can produce images with
    corresponding labels. The text that is generated is pretty
    much gibberish; randomly chosen characters (from a predefined
    set of characters).

    Args:
        count (int):

    Returns:
        GeneratorFromRandom
    """
    font_path = os.path.join(PROJECT_PATH, 'resources/fonts/Vudotronic.otf')
    background_images_dir = os.path.join(PROJECT_PATH, 'resources/background_images')
    gen = GeneratorFromRandom(count=count,
                              length=2,
                              allow_variable=True,
                              use_letters=True,
                              use_numbers=True,
                              use_symbols=True,
                              character_spacing=5,  # how much space to put between characters from one word
                              fonts=[font_path],
                              background_type=-1,  # means use images as background
                              image_dir=background_images_dir)
    return gen


def iterate_over_images_and_store(generator, images_dir):
    """Iterates over the given data generator, stores the images
    and stores the corresponding labels in a dataframe.

    Args:
        generator (GeneratorFromDict):
        images_dir (str):

    Returns:
        pd.DataFrame
    """
    os.makedirs(images_dir, exist_ok=True)
    data = pd.DataFrame(columns=['filename', 'label'])
    for img, label in tqdm(generator):
        random_id = uuid1()
        filename = "{}.jpg".format(random_id, label)
        path = os.path.join(images_dir, filename)
        img.save(path, "JPEG")
        data = data.append({'filename': filename, 'label': label}, ignore_index=True)
    return data


def store_df(df, output_dir, output_filename):
    """Stores a data frame in the given output directory
    under the given filename.

    Args:
        df (pd.DataFrame):
        output_dir (str):
        output_filename (str):
    """
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, output_filename), index=False)


def generate_data(nr_examples, output_filename):
    """Orchestrates the generation of one data set.

    Args:
        nr_examples (int):
        output_filename (str):
    """
    generator = create_random_image_generator(nr_examples)
    df = iterate_over_images_and_store(generator, known_args.images_dir)
    store_df(df, known_args.df_dir, output_filename)


def main():
    """Generates data for all three data sets: train, validation and test.
    """
    generate_data(known_args.train_size, 'train.csv')
    generate_data(known_args.validation_size, 'validation.csv')
    generate_data(known_args.test_size, 'test.csv')


if __name__ == '__main__':
    main()
