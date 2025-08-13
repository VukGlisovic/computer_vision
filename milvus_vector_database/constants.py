import os


# project information
PROJECT_NAME = 'milvus_vector_database'
PROJECT_PATH = os.path.join(os.path.realpath(__file__)[:os.path.realpath(__file__).find('/' + PROJECT_NAME)], PROJECT_NAME)


DB_NAME = 'glove.db'
COLLECTION_NAME = 'glove'
