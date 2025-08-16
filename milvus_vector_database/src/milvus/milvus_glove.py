from typing import Dict, Any, Union, List, Optional
import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np
from tqdm import tqdm
from pymilvus import MilvusClient

from milvus_vector_database.src.milvus.index_configuration import IndexOptimizationPresets, SearchConfig
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

    def drop_index(self, field_name: str = "vector") -> None:
        """Drop the current index."""
        try:
            self.client.drop_index(
                collection_name=COLLECTION_GLOVE,
                index_name=f"{field_name}_index"
            )
            self.current_index_config = None
            logger.info("Index dropped successfully")
        except Exception as e:
            logger.warning(f"Failed to drop index: {e}")

    def create_index(self, index_config: IndexConfig, field_name: str = "vector") -> None:
        """Create an optimized index for faster vector searches."""
        logger.info(f"Creating {index_config.index_type.value} index with metric {index_config.metric_type}")

        # Prepare index parameters
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name=field_name,
            index_type=index_config.index_type.value,
            index_name=f"{field_name}_index",
            metric_type=index_config.metric_type,
            params=index_config.params
        )

        # Create the index
        self.client.create_index(
            collection_name=COLLECTION_GLOVE,
            index_params=index_params
        )

        self.current_index_config = index_config
        logger.info(f"Index created successfully with parameters: {index_config.params}")

    def optimize_with_preset(self, preset_name: str) -> None:
        """Apply a predefined optimization preset."""
        presets = {
            "speed": IndexOptimizationPresets.speed_optimized(),
            "memory": IndexOptimizationPresets.memory_optimized(),
            "balanced": IndexOptimizationPresets.balanced(),
            "accuracy": IndexOptimizationPresets.accuracy_first(),
            "scann": IndexOptimizationPresets.scann_optimized(),
            "gpu": IndexOptimizationPresets.gpu_accelerated(),
        }

        if preset_name not in presets:
            raise ValueError(f"Unknown preset: {preset_name}. Available: {list(presets.keys())}")

        # Drop existing index if any
        self.drop_index()

        # Create new optimized index
        config = presets[preset_name]
        self.create_index(config)
        logger.info(f"Applied {preset_name} optimization preset")

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

    def search_vectors(self, query_vector: np.ndarray, filter: str = None, k: int = 1,
                       search_config: Optional[SearchConfig] = None) -> Dict[str, Any]:
        """Search for similar vectors with optional optimization parameters."""
        search_params = {}

        # Set search parameters based on current index type
        if self.current_index_config and search_config:
            search_params = search_config.params
        elif self.current_index_config:
            # Use default search parameters for the current index type
            search_params = self._get_default_search_params(self.current_index_config.index_type)

        res = self.client.search(
            collection_name=COLLECTION_GLOVE,
            data=query_vector,
            filter=filter,
            limit=k,
            output_fields=["id", "vector"],
            search_params=search_params if search_params else None
        )
        return res

    def _get_default_search_params(self, index_type: IndexType) -> Dict[str, Any]:
        """Get default search parameters for each index type."""
        default_search_params = {
            IndexType.FLAT: {},
            IndexType.IVF_FLAT: {"nprobe": 10},
            IndexType.IVF_PQ: {"nprobe": 10},
            IndexType.IVF_SQ8: {"nprobe": 10},
            IndexType.HNSW: {"ef": 64},
            IndexType.SCANN: {},
            IndexType.GPU_IVF_FLAT: {"nprobe": 10},
            IndexType.GPU_IVF_PQ: {"nprobe": 10},
        }
        return default_search_params.get(index_type, {})

    def query_vectors_by_ids(self, ids: Union[List[int], int]) -> Dict[str, Any]:
        res = self.client.query(
            collection_name=COLLECTION_GLOVE,
            ids=ids,
            output_fields=["id", "vector"],
        )
        return res
