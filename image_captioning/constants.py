import os
import tensorflow as tf


# project information
PROJECT_NAME = 'image_captioning'
PROJECT_PATH = os.path.join(os.path.realpath(__file__)[:os.path.realpath(__file__).find('/' + PROJECT_NAME)], PROJECT_NAME)

DATA_DIR = os.path.join(PROJECT_PATH, 'data')
