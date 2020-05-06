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
parser.add_argument('-lr', '--learning_rate',
                    type=float,
                    default=0.02,
                    help="Learning rate for Adam optimizer.")
known_args, _ = parser.parse_known_args()
logging.info('Content image path: %s', known_args.content_path)
logging.info('Style image path: %s', known_args.style_path)
logging.info('Image output path: %s', known_args.output_path)
logging.info('Alpha: %s', known_args.alpha)
logging.info('Beta: %s', known_args.beta)
logging.info('Learning rate: %s', known_args.learning_rate)


def load_images(content_path, style_path):
    """Loads a content and a style image.

    Args:
        content_path (str):
        style_path (str):

    Returns:
        tuple[tf.Tensor, tf.Tensor]
    """
    content_image = img_utils.load_img(content_path)
    style_image = img_utils.load_img(style_path)
    return content_image, style_image


@tf.function()
def train_step(image, content_targets, style_targets, extractor, optimizer):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content.style_content_loss(outputs, content_targets, style_targets, known_args.alpha, known_args.beta)

    grad = tape.gradient(loss, image)
    optimizer.apply_gradients([(grad, image)])
    image.assign(style_content.clip_0_1(image))


def create_model():
    content_layers = ['block5_conv2']
    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1']
    extractor = StyleContentModelVGG(style_layers, content_layers)
    return extractor


def main():
    content_image, style_image = load_images(known_args.content_path, known_args.style_path)
    extractor = create_model()
    style_targets = extractor(style_image)['style']
    content_targets = extractor(content_image)['content']
    generated_image = tf.Variable(content_image)

    opt = tf.optimizers.Adam(learning_rate=known_args.learning_rate, beta_1=0.99, epsilon=1e-1)

    start = time.time()

    epochs = 100
    steps_per_epoch = 5

    step = 0
    for n in range(epochs):
        for m in range(steps_per_epoch):
            step += 1
            train_step(generated_image, content_targets, style_targets, extractor, opt)
        output_path = ('_ep'+str(n).zfill(3)+'.').join(known_args.output_path.rsplit('.', 1))
        img_utils.store_tensor_image(generated_image, output_path)
        logging.info("Train step: {}".format(step))

    end = time.time()
    logging.info("Total time: {:.1f}".format(end - start))


if __name__ == '__main__':
    main()
