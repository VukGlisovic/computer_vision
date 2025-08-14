from typing import Dict, Any, Union, List, Optional
import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np
from tqdm import tqdm
from pymilvus import MilvusClient

from milvus_vector_database.constants import *


logger = logging.getLogger(__name__)


class IndexType(Enum):
    """Supported index types for vector optimization."""
    FLAT = "FLAT"           # Exact search, no optimization
    IVF_FLAT = "IVF_FLAT"   # Inverted file index with flat quantization
    IVF_PQ = "IVF_PQ"       # Inverted file with product quantization
    IVF_SQ8 = "IVF_SQ8"     # Inverted file with scalar quantization
    HNSW = "HNSW"           # Hierarchical navigable small world
    SCANN = "SCANN"         # Google's ScaNN algorithm
    GPU_IVF_FLAT = "GPU_IVF_FLAT"   # GPU-accelerated IVF_FLAT
    GPU_IVF_PQ = "GPU_IVF_PQ"       # GPU-accelerated IVF_PQ


@dataclass
class IndexConfig:
    """Configuration for vector index optimization."""
    index_type: IndexType
    metric_type: str = "L2"  # L2, IP (inner product), COSINE
    params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.params is None:
            self.params = self._get_default_params()
    
    def _get_default_params(self) -> Dict[str, Any]:
        """Get default parameters for each index type."""
        default_params = {
            IndexType.FLAT: {},
            IndexType.IVF_FLAT: {"nlist": 128},
            IndexType.IVF_PQ: {"nlist": 128, "m": 16, "nbits": 8},
            IndexType.IVF_SQ8: {"nlist": 128},
            IndexType.HNSW: {"M": 16, "efConstruction": 200},
            IndexType.SCANN: {"with_raw_data": True},
            IndexType.GPU_IVF_FLAT: {"nlist": 128},
            IndexType.GPU_IVF_PQ: {"nlist": 128, "m": 16, "nbits": 8},
        }
        return default_params.get(self.index_type, {})


@dataclass 
class SearchConfig:
    """Configuration for search parameters."""
    params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.params is None:
            self.params = {}


class OptimizationPreset:
    """Predefined optimization presets for common scenarios."""
    
    @staticmethod
    def speed_optimized() -> IndexConfig:
        """Optimized for fastest search speed (may sacrifice some accuracy)."""
        return IndexConfig(
            index_type=IndexType.HNSW,
            metric_type="L2",
            params={"M": 32, "efConstruction": 200}
        )
    
    @staticmethod
    def memory_optimized() -> IndexConfig:
        """Optimized for minimal memory usage."""
        return IndexConfig(
            index_type=IndexType.IVF_PQ,
            metric_type="L2", 
            params={"nlist": 256, "m": 8, "nbits": 8}
        )
    
    @staticmethod
    def balanced() -> IndexConfig:
        """Balanced speed and memory usage."""
        return IndexConfig(
            index_type=IndexType.IVF_SQ8,
            metric_type="L2",
            params={"nlist": 128}
        )
    
    @staticmethod
    def accuracy_first() -> IndexConfig:
        """Prioritizes accuracy over speed."""
        return IndexConfig(
            index_type=IndexType.FLAT,
            metric_type="L2"
        )
    
    @staticmethod
    def scann_optimized() -> IndexConfig:
        """Google ScaNN for large-scale similarity search."""
        return IndexConfig(
            index_type=IndexType.SCANN,
            metric_type="L2",
            params={"with_raw_data": True}
        )
    
    @staticmethod
    def gpu_accelerated() -> IndexConfig:
        """GPU-accelerated search for maximum performance."""
        return IndexConfig(
            index_type=IndexType.GPU_IVF_FLAT,
            metric_type="L2",
            params={"nlist": 128}
        )


class MilvusGlove:

    def __init__(self):
        self.client = MilvusClient(DB_PATH)
        self.vector_dim = 100
        self.current_index_config: Optional[IndexConfig] = None

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
            collection_name=COLLECTION_NAME,
            index_params=index_params
        )

        self.current_index_config = index_config
        logger.info(f"Index created successfully with parameters: {index_config.params}")

    def drop_index(self, field_name: str = "vector") -> None:
        """Drop the current index."""
        try:
            self.client.drop_index(
                collection_name=COLLECTION_NAME,
                index_name=f"{field_name}_index"
            )
            self.current_index_config = None
            logger.info("Index dropped successfully")
        except Exception as e:
            logger.warning(f"Failed to drop index: {e}")

    def get_index_info(self) -> Dict[str, Any]:
        """Get information about current indexes."""
        try:
            indexes = self.client.list_indexes(collection_name=COLLECTION_NAME)
            return {"indexes": indexes, "current_config": self.current_index_config}
        except Exception as e:
            logger.error(f"Failed to get index info: {e}")
            return {"indexes": [], "current_config": self.current_index_config}

    def optimize_with_preset(self, preset_name: str) -> None:
        """Apply a predefined optimization preset."""
        presets = {
            "speed": OptimizationPreset.speed_optimized(),
            "memory": OptimizationPreset.memory_optimized(),
            "balanced": OptimizationPreset.balanced(),
            "accuracy": OptimizationPreset.accuracy_first(),
            "scann": OptimizationPreset.scann_optimized(),
            "gpu": OptimizationPreset.gpu_accelerated(),
        }

        if preset_name not in presets:
            raise ValueError(f"Unknown preset: {preset_name}. Available: {list(presets.keys())}")

        # Drop existing index if any
        self.drop_index()

        # Create new optimized index
        config = presets[preset_name]
        self.create_index(config)
        logger.info(f"Applied {preset_name} optimization preset")

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
            collection_name=COLLECTION_NAME,
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
            collection_name=COLLECTION_NAME,
            ids=ids,
            output_fields=["id", "vector"],
        )
        return res
