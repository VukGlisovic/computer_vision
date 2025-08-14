from typing import Dict, Any, Union, List, Optional
import logging

import numpy as np
from tqdm import tqdm
from pymilvus import MilvusClient

from milvus_vector_database.constants import *


logger = logging.getLogger(__name__)


class MilvusGlove:

	def __init__(self):
		self.client = MilvusClient(DB_PATH)
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
	
	def insert_vectors(self, vectors: np.ndarray, ids: np.ndarray, chunk_size: int = 10000, timeout: int = 10) -> None:
		logger.info(f"Inserting vectors into collection '{COLLECTION_NAME}' in chunks of {chunk_size} vectors.")
		for i in tqdm(range(0, len(vectors), chunk_size), desc="Inserting vectors"):
			chunk_vectors = vectors[i:i+chunk_size]
			chunk_ids = ids[i:i+chunk_size]
			self.insert_vectors_chunk(chunk_vectors, chunk_ids, timeout)
	
	def insert_vectors_chunk(self, vectors: np.ndarray, ids: np.ndarray, timeout: int = 10) -> Dict[str, Any]:
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
		return res
	
	def search_vectors(self, query_vector: np.ndarray, k: int = 10) -> Dict[str, Any]:
		res = self.client.search(
			collection_name=COLLECTION_NAME,
			data=query_vector,
			limit=k,
			output_fields=["id", "vector"]
		)
		return res

	def query_vectors_by_ids(self, ids: Union[List[int], int]) -> Dict[str, Any]:
		res = self.client.query(
			collection_name=COLLECTION_NAME,
			ids=ids,
			output_fields=["id", "vector"],
		)
		return res
