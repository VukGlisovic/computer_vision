import os


# project information
PROJECT_NAME = 'blazeface'
PROJECT_PATH = os.path.join(os.path.realpath(__file__)[:os.path.realpath(__file__).find('/' + PROJECT_NAME)], PROJECT_NAME)

DATA_DIR = os.path.join(PROJECT_PATH, 'data')

IMG_SIZE = 128
BBOX_PADDING = 1e-3

