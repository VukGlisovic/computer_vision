import os


# project information
PROJECT_NAME = 'milvus_vector_database'
PROJECT_PATH = os.path.join(os.path.realpath(__file__)[:os.path.realpath(__file__).find('/' + PROJECT_NAME)], PROJECT_NAME)


# For Milvus Lite (no docker container needed)
DB_PATH = os.path.join(PROJECT_PATH, 'data/glove.db')
# For Milvus remote from docker container
DB_URI = 'http://localhost:19530'

COLLECTION_NAME = 'glove'
