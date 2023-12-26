import os


PROJECT_NAME = 'llama'
PROJECT_PATH = os.path.join(os.path.realpath(__file__)[:os.path.realpath(__file__).find('/' + PROJECT_NAME)], PROJECT_NAME)

DATA_DIR = os.path.join(PROJECT_PATH, 'data')
EXPERIMENT_DIR = os.path.join(DATA_DIR, 'experiment')
