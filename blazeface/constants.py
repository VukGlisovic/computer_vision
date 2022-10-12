import os


# project information
PROJECT_NAME = 'blazeface'
PROJECT_PATH = os.path.join(os.path.realpath(__file__)[:os.path.realpath(__file__).find('/' + PROJECT_NAME)], PROJECT_NAME)

DATA_DIR = os.path.join(PROJECT_PATH, 'data')

IMG_SIZE = 128
BBOX_PADDING = 1e-3
BOX_VARIANCE = [0.1, 0.1, 0.2, 0.2]
N_LANDMARKS = 6
FMAPS_WITH_ANCHORS = [16, 8]  # which feature map sizes have anchors
N_ANCHORS_PER_LOC = [2, 6]  # number of anchor predictions per location in the 16x16 and 8x8 feature maps respectively
assert len(FMAPS_WITH_ANCHORS) == len(N_ANCHORS_PER_LOC)
