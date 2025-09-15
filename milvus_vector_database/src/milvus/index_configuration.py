from typing import Dict, Any, Union, List, Optional
import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np
from tqdm import tqdm


logger = logging.getLogger(__name__)


class IndexType(Enum):
    """Supported index types for vector optimization from this enum.

    Pymilvus also has `from pymilvus import IndexType`, but for some reason I couldn't
    find SCANN even though it's documented on:
    https://milvus.io/api-reference/pymilvus/v2.6.x/MilvusClient/Collections/IndexType.md
    """
    FLAT = "FLAT"           # Exact search, no optimization
    IVF_FLAT = "IVF_FLAT"   # Inverted file index with flat quantization
    IVF_PQ = "IVF_PQ"       # Inverted file with product quantization
    IVF_SQ8 = "IVF_SQ8"     # Inverted file with scalar quantization
    HNSW = "HNSW"           # Hierarchical navigable small world
    SCANN = "SCANN"         # Google's ScaNN algorithm
    GPU_BRUTE_FORCE = "GPU_BRUTE_FORCE"   # GPU-accelerated IVF_FLAT
    GPU_IVF_FLAT = "GPU_IVF_FLAT"         # GPU-accelerated IVF_FLAT
    GPU_IVF_PQ = "GPU_IVF_PQ"             # GPU-accelerated IVF_PQ


@dataclass
class IndexConfig:
    index_type: IndexType
    metric_type: str = "COSINE"  # COSINE, L2, IP (inner product)
    params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.params is None:
            # If no params are provided, we'll take the default values
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
            IndexType.GPU_BRUTE_FORCE: {},
            IndexType.GPU_IVF_FLAT: {"nlist": 128},
            IndexType.GPU_IVF_PQ: {"nlist": 128, "m": 16, "nbits": 8},
        }
        params = default_params.get(self.index_type, {})
        logger.info(f"No params configured for index type {self.index_type}. Loading default params: {params}.")
        return params


class IndexOptimizationPresets:

    @classmethod
    def get_preset(cls, preset_name):
        presets = {
            "speed": cls.speed_optimized(),
            "memory": cls.memory_optimized(),
            "balanced": cls.balanced(),
            "accuracy": cls.accuracy_first(),
            "scann": cls.scann_optimized(),
            "gpu": cls.gpu_accelerated(),
            "accuracy_gpu": cls.accuracy_first_gpu(),
        }

        if preset_name not in presets:
            raise ValueError(f"Unknown preset: {preset_name}. Available presets: {list(presets.keys())}")

        return presets[preset_name]
    
    @staticmethod
    def speed_optimized() -> IndexConfig:
        """
        Optimized for fastest search speed (may sacrifice some accuracy).

        Uses HNSW (Hierarchical Navigable Small World) - a graph-based algorithm that 
        creates a multi-layer network of connections between vectors. It provides very 
        fast approximate searches by navigating through the graph hierarchy, starting 
        from the top layer and drilling down. The higher M=32 creates more connections 
        for better speed.
        """
        return IndexConfig(
            index_type=IndexType.HNSW,
            metric_type="COSINE",
            params={"M": 32, "efConstruction": 200}
        )
    
    @staticmethod
    def memory_optimized() -> IndexConfig:
        """
        Optimized for minimal memory usage.
        
        Uses IVF_PQ (Inverted File with Product Quantization) - first partitions vectors 
        into clusters (nlist=256), then compresses each vector using product quantization 
        (dividing into m=8 subvectors, each quantized to 8 bits). This dramatically reduces 
        memory usage by storing compressed representations instead of full vectors.
        """
        # 256 clusters, 10 subvectors, 8 bits per subvector
        return IndexConfig(
            index_type=IndexType.IVF_PQ,
            metric_type="COSINE", 
            params={"nlist": 256, "m": 10, "nbits": 8}
        )
    
    @staticmethod
    def balanced() -> IndexConfig:
        """
        Balanced speed and memory usage.
        
        Uses IVF_SQ8 (Inverted File with Scalar Quantization 8-bit) - partitions vectors 
        into clusters like IVF_FLAT, but compresses each vector component from 32-bit floats 
        to 8-bit integers. This provides a good balance between memory savings and search 
        accuracy.
        """
        return IndexConfig(
            index_type=IndexType.IVF_SQ8,
            metric_type="COSINE",
            params={"nlist": 256}
        )
    
    @staticmethod
    def accuracy_first() -> IndexConfig:
        """
        Prioritizes accuracy over speed.
        
        Uses FLAT - performs exact brute-force search by computing distances to all vectors 
        in the collection. No approximation means perfect accuracy, but it's the slowest 
        option as dataset size grows.
        """
        return IndexConfig(
            index_type=IndexType.FLAT,
            metric_type="COSINE",
            params={}
        )
    
    @staticmethod
    def scann_optimized() -> IndexConfig:
        """
        Google ScaNN for large-scale similarity search.
        
        Uses SCANN (Scalable Nearest Neighbors) - Google's algorithm that combines learned 
        quantization with efficient search techniques. It's designed for large-scale similarity 
        search and can handle billions of vectors efficiently while maintaining good accuracy.
        """
        return IndexConfig(
            index_type=IndexType.SCANN,
            metric_type="COSINE",
            params={"with_raw_data": True}
        )
    
    @staticmethod
    def gpu_accelerated() -> IndexConfig:
        """
        GPU-accelerated search for maximum performance.
        
        Uses GPU_IVF_FLAT - same as IVF_FLAT (partitioning vectors into clusters for faster 
        search) but accelerated using GPU computing. This leverages parallel processing 
        capabilities of GPUs for much faster search performance on large datasets.
        """
        return IndexConfig(
            index_type=IndexType.GPU_IVF_FLAT,
            metric_type="COSINE",
            params={"nlist": 256}
        )

    @staticmethod
    def accuracy_first_gpu() -> IndexConfig:
        """
        Prioritizes accuracy over speed.

        Basically the same as FLAT indexing, but then using GPU - performs exact brute-force
        search by computing distances to all vectors in the collection. No approximation means
        perfect accuracy. Relatively slow, but the GPU may compensate some speed.
        """
        return IndexConfig(
            index_type=IndexType.GPU_BRUTE_FORCE,
            metric_type="COSINE",
            params={}
        )


@dataclass
class SearchConfig:
    """Configuration for search parameters."""
    params: Dict[str, Any] = None

    def __post_init__(self):
        if self.params is None:
            self.params = {}

    def set_default_search_params(self, index_type: IndexType) -> None:
        """Get default search parameters for each index type."""
        default_search_params = {
            IndexType.FLAT: {},
            IndexType.IVF_FLAT: {"nprobe": 10},
            IndexType.IVF_PQ: {"nprobe": 10},
            IndexType.IVF_SQ8: {"nprobe": 10},
            IndexType.HNSW: {"ef": 64},
            IndexType.SCANN: {},
            IndexType.GPU_BRUTE_FORCE: {},
            IndexType.GPU_IVF_FLAT: {"nprobe": 10},
            IndexType.GPU_IVF_PQ: {"nprobe": 10},
        }
        self.params = default_search_params.get(index_type, {})
        logger.info(f"Set default search params for index_type {index_type}: {self.params}")
