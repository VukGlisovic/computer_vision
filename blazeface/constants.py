import os


# project information
PROJECT_NAME = 'blazeface'
PROJECT_PATH = os.path.join(os.path.realpath(__file__)[:os.path.realpath(__file__).find('/' + PROJECT_NAME)], PROJECT_NAME)

DATA_DIR = os.path.join(PROJECT_PATH, 'data')

# architecture hyperparameters
IMG_SIZE = 128
BBOX_PADDING = 1e-3
N_LANDMARKS = 6
IOU_THRESHOLD = 0.5
FMAPS_WITH_ANCHORS = [16, 8]  # which feature map sizes have anchors
N_ANCHORS_PER_LOC = [2, 6]  # number of anchor predictions per location in the 16x16 and 8x8 feature maps respectively
assert len(FMAPS_WITH_ANCHORS) == len(N_ANCHORS_PER_LOC)
BOX_VARIANCE = [0.1, 0.1, 0.2, 0.2]
LMARK_VARIANCE = N_LANDMARKS * BOX_VARIANCE[0:2]

# loss function hyperparameters
NEG_POS_RATIO = 3
LOC_LOSS_ALPHA = 1
