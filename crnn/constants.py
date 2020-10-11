import os


PROJECT_NAME = 'crnn'
PROJECT_PATH = os.path.join(os.path.realpath(__file__).split(PROJECT_NAME)[0], PROJECT_NAME)

IM_WIDTH = 512
IM_HEIGHT = 32

MAX_CHARACTERS = 40
CLASSES_PATH = os.path.join(PROJECT_PATH, 'config/characters.txt')
BLANK_INDEX = 92

DIR_IMAGES = os.path.join(PROJECT_PATH, 'data/images/')

FILENAME = 'filename'
LABEL = 'label'
