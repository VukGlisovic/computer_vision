import sys
import time
import logging
import argparse
import tensorflow as tf

from neural_style_transfer.src.model import StyleContentModelVGG
from neural_style_transfer.src.utils import image as img_utils
from neural_style_transfer.src import style_content

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
                    help="Where to store the generated image. Note that the epoch number"
                         "will be appended to the filename (before the extension).")
parser.add_argument('-md', '--max_dim',
                    type=int,
                    default=99999,
                    help="The max size of the longest image dimension for both the content"
                         "and the style image. If the height or width exceeds max_dim, then"
                         "the image is resized to agree with the max_dim.")
parser.add_argument('-a', '--alpha',
                    type=float,
                    default=1000,
                    help="How much to weight the content loss.")
parser.add_argument('-b', '--beta',
                    type=float,
                    default=1,
                    help="How much to weight the style loss.")
parser.add_argument('-vw', '--variation_weight',
                    type=float,
                    default=3,
                    help="How much to weight the variational loss.")
parser.add_argument('-lr', '--learning_rate',
                    type=float,
                    default=0.02,
                    help="Learning rate for Adam optimizer.")
parser.add_argument('-ep', '--epochs',
                    type=int,
                    default=100,
                    help="Number of epochs. At the end of each epoch, the intermediate generated"
                         "image will be stored.")
parser.add_argument('-st', '--steps_per_epoch',
                    type=int,
                    default=10,
                    help="The number of steps to take per epoch.")
parser.add_argument('-wn', '--initialize_with_noise',
                    action='store_true',
                    help="Whether to initialize the generated image from the content image"
                         "with or without noise.")
known_args, _ = parser.parse_known_args()
logging.info('Content image path: %s', known_args.content_path)
logging.info('Style image path: %s', known_args.style_path)
logging.info('Image output path: %s', known_args.output_path)
logging.info('Max dimension: %s', known_args.max_dim)
logging.info('Alpha: %s', known_args.alpha)
logging.info('Beta: %s', known_args.beta)
logging.info('Image variation weight: %s', known_args.variation_weight)
logging.info('Learning rate: %s', known_args.learning_rate)
logging.info('Epochs: %s', known_args.epochs)
logging.info('Steps per epoch: %s', known_args.steps_per_epoch)
logging.info('Inialize with noise: %s', known_args.initialize_with_noise)


def load_images(content_path, style_path, max_dim):
    """Loads a content and a style image.

    Args:
        content_path (str):
        style_path (str):
        max_dim (int):

    Returns:
        tuple[tf.Tensor, tf.Tensor]
    """
    content_image = img_utils.load_img(content_path, max_dim)
    logging.info("Shape content image: %s", content_image.get_shape())
    style_image = img_utils.load_img(style_path, max_dim)
    logging.info("Shape style image: %s", style_image.get_shape())
    return content_image, style_image


@tf.function()
def train_step(generated_img, content_targets, style_targets, extractor, optimizer):
    """Calculates gradients with respect to the generated image pixels
    and applies the update to the generated image pixels.

    Args:
        generated_img (tf.Tensor): the generated image
        content_targets (dict): {'content': content_dict, 'style': style_dict}
            for the content image.
        style_targets: {'content': content_dict, 'style': style_dict} for the
            style image.
        extractor (StyleContentModelVGG):
        optimizer (tf.optimizers.Adam):
    """
    with tf.GradientTape() as tape:
        generated_img_targets = extractor(generated_img)
        loss = style_content.style_content_loss(generated_img_targets, content_targets, style_targets, known_args.alpha, known_args.beta) \
            + style_content.image_variation(generated_img, known_args.variation_weight)

    grad = tape.gradient(loss, generated_img)
    optimizer.apply_gradients([(grad, generated_img)])
    generated_img.assign(style_content.clip_0_1(generated_img))
    return loss


def create_model():
    """Creates a model by extracting the needed layers from a
    vgg network.

    Returns:
        StyleContentModelVGG
    """
    content_layers = ['block4_conv2']
    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1']
    extractor = StyleContentModelVGG(style_layers, content_layers)
    return extractor


def main():
    """Sets up and executes the training.
    """
    content_image, style_image = load_images(known_args.content_path, known_args.style_path, known_args.max_dim)
    extractor = create_model()
    content_targets = extractor(content_image)['content']  # dict with vgg content and style values for content image
    style_targets = extractor(style_image)['style']  # dict with vgg content and style values for style image
    if known_args.initialize_with_noise:
        generated_image = tf.Variable(
            tf.random.uniform(content_image.get_shape(), minval=0.0, maxval=1.0, dtype=content_image.dtype, seed=42)
        )
    else:
        generated_image = tf.Variable(content_image)  # initialize generated image from content image
    img_utils.store_tensor_image(generated_image, known_args.output_path)

    opt = tf.optimizers.Adam(learning_rate=known_args.learning_rate, beta_1=0.99, epsilon=1e-1)

    start = time.time()

    step = 0
    total_loss = None
    for ep in range(known_args.epochs):
        for st in range(known_args.steps_per_epoch):
            step += 1
            total_loss = train_step(generated_image, content_targets, style_targets, extractor, opt)
        output_path = ('_ep'+str(ep).zfill(3)+'.').join(known_args.output_path.rsplit('.', 1))
        img_utils.store_tensor_image(generated_image, output_path)
        logging.info("Train step: %s. Loss %s", step, total_loss.numpy())

    end = time.time()
    logging.info("Total time: {:.1f}".format(end - start))


if __name__ == '__main__':
    main()
