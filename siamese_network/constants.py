import os


# project information
PROJECT_NAME = 'siamese_network'
PROJECT_PATH = os.path.join(os.path.realpath(__file__)[:os.path.realpath(__file__).find('/' + PROJECT_NAME)], PROJECT_NAME)


IMG_SPATIAL_SIZE = (200, 200)
N_CHANNELS = 3
