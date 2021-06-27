import os
import tensorflow as tf


# project information
PROJECT_NAME = 'efficientdet'
PROJECT_PATH = os.path.join(os.path.realpath(__file__)[:os.path.realpath(__file__).find('/' + PROJECT_NAME)], PROJECT_NAME)


IMG_SHAPE = (512, 512, 1)
BOX_VARIANCE = tf.constant([0.1, 0.1, 0.2, 0.2], dtype=tf.float32)
