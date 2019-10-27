import os


PROJECT_NAME = 'unet'
PROJECT_PATH = os.path.join(os.path.realpath(__file__).split(PROJECT_NAME)[0], PROJECT_NAME)

IM_WIDTH = 101
IM_HEIGHT = 101
IM_CHANNELS = 1
IMAGE_SHAPE = (IM_WIDTH, IM_HEIGHT, IM_CHANNELS)

CHECKPOINT_DIR = os.path.join(PROJECT_PATH, 'model/unet_saved_model')
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "cp-{epoch:04d}.ckpt")
