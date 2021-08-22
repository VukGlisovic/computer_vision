import os
import tensorflow as tf


# project information
PROJECT_NAME = 'efficientdet'
PROJECT_PATH = os.path.join(os.path.realpath(__file__)[:os.path.realpath(__file__).find('/' + PROJECT_NAME)], PROJECT_NAME)


IMG_SHAPE = (512, 512, 1)

ASPECT_RATIOS = tf.cast([0.5, 1.0, 2.0], tf.float32)
SCALES = tf.cast([2 ** x for x in [0, 1 / 3, 2 / 3]], tf.float32)
NUM_ANCHORS = len(ASPECT_RATIOS) * len(SCALES)
BOX_VARIANCE = tf.constant([0.1, 0.1, 0.2, 0.2], dtype=tf.float32)
