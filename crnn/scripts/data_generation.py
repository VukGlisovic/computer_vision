import os
import sys
import logging
import argparse
from trdg.generators import GeneratorFromDict
# from crnn.vendor.TextRecognitionDataGenerator.trdg.generators import GeneratorFromDict

logformat = '%(asctime)s | %(levelname)s | [%(filename)s:%(lineno)s - %(funcName)s] %(message)s'
logging.basicConfig(format=logformat, level=logging.INFO, stream=sys.stdout)

parser = argparse.ArgumentParser()
parser.add_argument('--nr_train_images',
                    type=int,
                    default=1000,
                    help="The number of images to generate for training.")
parser.add_argument('--nr_validation_images',
                    type=int,
                    default=250,
                    help="The number of images to generate for validation.")
parser.add_argument('--nr_test_images',
                    type=int,
                    default=250,
                    help="The number of images to generate for testing.")
parser.add_argument('--images_dir',
                    type=str,
                    default='../data/images',
                    help="Where to store train/validation/test images.")
known_args, _ = parser.parse_known_args()
logging.info("Given input parameters:\n%s", "\n".join(["{}: {}".format(k, v) for k,v in known_args.__dict__.items()]))


def create_random_image_generator():
    gen = GeneratorFromDict()
    return gen


def iterate_over_images_and_store(generator, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for img, label in generator:
        path = os.path.join(output_dir, "{}.jpg".format(label))
        img.save(path, "JPEG")


def main():
    generator = create_random_image_generator()
    iterate_over_images_and_store(generator, known_args.images_dir)

if __name__ == '__main__':
    main()
