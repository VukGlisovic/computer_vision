from pymilvus import MilvusClient

from milvus_vector_database.constants import DB_NAME


class MilvusGlove:

	def __init__(self):
		self.client = MilvusClient(DB_NAME)
