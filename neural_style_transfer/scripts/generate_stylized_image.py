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
parser.add_argument('-a', '--alpha',
                    type=float,
                    default=1000,
                    help="How much to weight the content loss.")
parser.add_argument('-b', '--beta',
                    type=float,
                    default=0.01,
                    help="How much to weight the style loss.")
parser.add_argument('-vw', '--variation_weight',
                    type=float,
                    default=3,
                    help="How much to weight the variational loss.")
parser.add_argument('-lr', '--learning_rate',
                    type=float,
                    default=0.02,
                    help="Learning rate for Adam optimizer.")
parser.add_argument('-wn', '--initialize_with_noise',
                    action='store_true',
                    help="Whether to initialize the generated image from the content image"
                         "with or without noise.")
known_args, _ = parser.parse_known_args()
logging.info('Content image path: %s', known_args.content_path)
logging.info('Style image path: %s', known_args.style_path)
logging.info('Image output path: %s', known_args.output_path)
logging.info('Alpha: %s', known_args.alpha)
logging.info('Beta: %s', known_args.beta)
logging.info('Image variation weight: %s', known_args.variation_weight)
logging.info('Learning rate: %s', known_args.learning_rate)
logging.info('Inialize with noise: %s', known_args.initialize_with_noise)


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
    content_layers = ['block5_conv2']
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
    content_image, style_image = load_images(known_args.content_path, known_args.style_path)
    extractor = create_model()
    content_targets = extractor(content_image)['content']  # dict with vgg content and style values for content image
    style_targets = extractor(style_image)['style']  # dict with vgg content and style values for style image
    generated_image = tf.Variable(content_image)  # initialize generated image from content image
    if known_args.initialize_with_noise:
        generated_image = tf.add(
            generated_image,
            tf.random.uniform(generated_image.get_shape(), minval=-0.5, maxval=0.5, dtype=generated_image.dtype, seed=42)
        )

    opt = tf.optimizers.Adam(learning_rate=known_args.learning_rate, beta_1=0.99, epsilon=1e-1)

    start = time.time()

    epochs = 300
    steps_per_epoch = 3

    step = 0
    total_loss = None
    for ep in range(epochs):
        for st in range(steps_per_epoch):
            step += 1
            total_loss = train_step(generated_image, content_targets, style_targets, extractor, opt)
        output_path = ('_ep'+str(ep).zfill(3)+'.').join(known_args.output_path.rsplit('.', 1))
        img_utils.store_tensor_image(generated_image, output_path)
        logging.info("Train step: %s. Loss %s", step, total_loss.numpy())

    end = time.time()
    logging.info("Total time: {:.1f}".format(end - start))


if __name__ == '__main__':
    main()
