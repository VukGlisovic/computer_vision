import sys
import logging
import argparse
import tensorflow as tf
import tensorflow_hub as hub
from neural_style_transfer.src.utils import image as img_utils

logformat = '%(asctime)s | %(levelname)s | [%(filename)s:%(lineno)s - %(funcName)s] %(message)s'
logging.basicConfig(format=logformat, level=logging.INFO, stream=sys.stdout)


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--content_path',
                    type=str,
                    help="Where to find the content image.")
parser.add_argument('-s', '--style_path',
                    type=str,
                    help="Where to find the style image.")
parser.add_argument('-o', '--output_path',
                    type=str,
                    help="Where to store the stylized image.")
known_args, _ = parser.parse_known_args()
logging.info('Content image path: %s', known_args.content_path)
logging.info('Style image path: %s', known_args.style_path)
logging.info('Image output path: %s', known_args.output_path)


def load_images(content_path, style_path):
    """Loads a content and a style image.

    Args:
        content_path (str):
        style_path (str):

    Returns:
        tuple[tf.Tensor, tf.Tensor]
    """
    content_image = img_utils.load_img(content_path)
    logging.info("Shape content image: %s", content_image.get_shape())
    style_image = img_utils.load_img(style_path)
    logging.info("Shape style image: %s", style_image.get_shape())
    return content_image, style_image


def tf_hub_style_transfer(content_image, style_image):
    """Loads an module from tensorflow hub that is already
    specialized for merging a content and style image into
    a stylized image.

    Args:
        content_image (tf.Tensor):
        style_image (tf.Tensor):

    Returns:
        tf.Tensor
    """
    hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/1')
    stylized_image = hub_module(tf.constant(content_image), tf.constant(style_image))[0]
    return stylized_image


def main():
    """Combines functionality.
    """
    c_img, s_img = load_images(known_args.content_path, known_args.style_path)
    result = tf_hub_style_transfer(c_img, s_img)
    img_utils.store_tensor_image(result, known_args.output_path)


if __name__ == '__main__':
    main()
