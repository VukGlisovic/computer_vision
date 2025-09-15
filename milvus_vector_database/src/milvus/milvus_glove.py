from typing import Dict, Any, Union, List, Optional
import logging
import time
from dataclasses import dataclass
from enum import Enum

import numpy as np
from tqdm import tqdm
from pymilvus import MilvusClient

from milvus_vector_database.src.milvus.index_configuration import IndexType, IndexConfig, IndexOptimizationPresets, SearchConfig
from milvus_vector_database.constants import *


logger = logging.getLogger(__name__)


class MilvusGlove:

    def __init__(self, remote=True):
        self.remote = remote

        db_uri = DB_URI if remote else DB_PATH
        self.client = MilvusClient(
            uri=db_uri
        )
        self.vector_dim = 100
        self.current_index_config: Optional[IndexConfig] = None
        self.search_config = SearchConfig()
    
    def create_collection(self, overwrite: bool = False):
        collection_exists = self.client.has_collection(collection_name=COLLECTION_GLOVE)
        if not overwrite and collection_exists:
            logger.info(f"Collection '{COLLECTION_GLOVE}' already exists, skipping creation.")
            return
        elif overwrite and collection_exists:
            logger.info(f"Dropping existing collection '{COLLECTION_GLOVE}'.")
            self.client.drop_collection(collection_name=COLLECTION_GLOVE)
        logger.info(f"Creating new collection '{COLLECTION_GLOVE}'.")
        self.client.create_collection(
            collection_name=COLLECTION_GLOVE,
            dimension=self.vector_dim
        )

    def drop_index(self, index_name: str) -> None:
        try:
            self.client.drop_index(
                collection_name=COLLECTION_GLOVE,
                index_name=index_name
            )
            self.current_index_config = None
            logger.info(f"Index '{index_name}' dropped successfully.")
        except Exception as e:
            logger.warning(f"Failed to drop index: {e}")

    def create_vector_index(self, index_config: IndexConfig) -> None:
        logger.info(f"Creating {index_config.index_type.name} index with metric {index_config.metric_type}")

        if index_config == self.current_index_config:
            logger.info(f"Index {index_config.index_type.name} already exists with the same parameters, skipping creation.")
            return

        field_name = 'vector'

        # In order to create a new index, we need to have the collection released from memory
        collection_state = self.client.get_load_state(COLLECTION_GLOVE)['state']
        if collection_state.name != 'NotLoad':
            self.client.release_collection(collection_name=COLLECTION_GLOVE)
            logger.info(f"Released collection '{COLLECTION_GLOVE}'.")

        # The vector field allows for only one index at a time
        current_index_names = [idx_name for idx_name in self.client.list_indexes(COLLECTION_GLOVE) if field_name in idx_name]
        for idx_name in current_index_names:
            self.drop_index(index_name=idx_name)

        # Prepare index parameters
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name=field_name,
            index_type=index_config.index_type.value,
            index_name=f"{field_name}_{index_config.index_type.name}_index",
            metric_type=index_config.metric_type,
            params=index_config.params
        )
        # Create the vector index
        logger.info("Starting index creation...")
        start_time = time.time()
        self.client.create_index(
            collection_name=COLLECTION_GLOVE,
            index_params=index_params
        )
        logger.info(f"Index creation completed in {time.time() - start_time:.1f} seconds")
        self.current_index_config = index_config
        logger.info(f"Index created successfully with parameters: {index_config.params}")

        # Load the collection back into memory
        self.client.load_collection('glove')
        logger.info(f"Loaded collection '{COLLECTION_GLOVE}'.")

    def insert_vectors_chunk(self, vectors: np.ndarray, ids: np.ndarray, timeout: int = 10) -> Dict[str, Any]:
        data = [
            {
                'id': id,
                'vector': vector
            }
            for id, vector in zip(ids, vectors)
        ]
        res = self.client.insert(
            collection_name=COLLECTION_GLOVE,
            data=data,
            timeout=timeout
        )
        return res

    def insert_vectors(self, vectors: np.ndarray, ids: np.ndarray, chunk_size: int = 10000, timeout: int = 10) -> None:
        logger.info(f"Inserting vectors into collection '{COLLECTION_GLOVE}' in chunks of {chunk_size} vectors.")
        for i in tqdm(range(0, len(vectors), chunk_size), desc="Inserting vectors"):
            chunk_vectors = vectors[i:i+chunk_size]
            chunk_ids = ids[i:i+chunk_size]
            self.insert_vectors_chunk(chunk_vectors, chunk_ids, timeout)
    
    def set_search_config(self, search_config: Optional[SearchConfig] = None) -> None:
        if search_config is None:
            search_config = SearchConfig()
            search_config.set_default_search_params(self.current_index_config.index_type)
            logger.info(f"No search config provided, using default search parameters for {self.current_index_config.index_type.name} index: {search_config.params}.")
        else:
            logger.info(f"Using provided search config: {search_config.params}.")
        self.search_config = search_config

    def search_vectors(self, query_vector: np.ndarray, filter: str = None, k: int = 1) -> Dict[str, Any]:
        """Search for similar vectors with optional optimization parameters."""
        res = self.client.search(
            collection_name=COLLECTION_GLOVE,
            data=query_vector,
            filter=filter,
            limit=k,
            output_fields=["id", "vector"],
            search_params=self.search_config.params
        )
        return res

    def query_vectors_by_ids(self, ids: Union[List[int], int]) -> Dict[str, Any]:
        res = self.client.query(
            collection_name=COLLECTION_GLOVE,
            ids=ids,
            output_fields=["id", "vector"],
        )
        return res
