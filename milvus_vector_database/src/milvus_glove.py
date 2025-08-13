from typing import Dict, Any
import logging

import numpy as np
from pymilvus import MilvusClient

from milvus_vector_database.constants import *


logger = logging.getLogger(__name__)


class MilvusGlove:

	def __init__(self):
		self.client = MilvusClient(DB_NAME)
		self.vector_dim = 100
	
	def create_collection(self, overwrite: bool = False):
		if not overwrite and self.client.has_collection(collection_name=COLLECTION_NAME):
			logger.info(f"Collection '{COLLECTION_NAME}' already exists, skipping creation.")
			return
		if overwrite and self.client.has_collection(collection_name=COLLECTION_NAME):
			logger.info(f"Dropping existing collection '{COLLECTION_NAME}'.")
			self.client.drop_collection(collection_name=COLLECTION_NAME)
		logger.info(f"Creating new collection '{COLLECTION_NAME}'.")
		self.client.create_collection(
			collection_name=COLLECTION_NAME,
			dimension=self.vector_dim
		)
	
	def insert_vectors(self, vectors: np.ndarray, ids: np.ndarray, timeout: int = 10) -> Dict[str, Any]:
		data = [
			{
				'id': id,
				'vector': vector
			}
			for id, vector in zip(ids, vectors)
		]
		res = self.client.insert(
			collection_name=COLLECTION_NAME, 
			data=data,
			timeout=timeout
		)
		logger.info(res)
		return res
	
	def search_vectors(self, query_vector: np.ndarray, k: int = 10):
		res = self.client.search(
			collection_name=COLLECTION_NAME,
			query_vector=query_vector,
			k=k)
