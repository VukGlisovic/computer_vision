import os


PROJECT_NAME = 'crnn'
PROJECT_PATH = os.path.join(os.path.realpath(__file__).split(PROJECT_NAME)[0], PROJECT_NAME)

IM_WIDTH = 512
IM_HEIGHT = 32
CHANNELS = 3

MAX_CHARACTERS = 40
CLASSES_PATH = os.path.join(PROJECT_PATH, 'config/characters.txt')
with open(CLASSES_PATH) as f:
    characters = f.read()
CHARACTERS = characters.split('\n')
NR_CHARACTERS = len(CHARACTERS)
BLANK_INDEX = NR_CHARACTERS - 1  # last row in the classes file represents the blank character

DIR_IMAGES = os.path.join(PROJECT_PATH, 'data/images/')
DIR_PROCESSED = os.path.join(PROJECT_PATH, 'data/processed/')

FILENAME = 'filename'
LABEL = 'label'
