import os


# project information
PROJECT_NAME = 'vision_transformer'
PROJECT_PATH = os.path.join(os.path.realpath(__file__)[:os.path.realpath(__file__).find('/' + PROJECT_NAME)], PROJECT_NAME)

EXPERIMENTS_DIR = os.path.join(PROJECT_PATH, 'experiments')

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
