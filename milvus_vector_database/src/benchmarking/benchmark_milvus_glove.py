"""
Benchmark script to compare performance of different Milvus optimization strategies.

This script tests various index types and quantization methods to help you choose
the best optimization strategy for your specific use case.
"""
import time
import logging
from typing import Dict, List, Tuple

import numpy as np
from datasets import Dataset
from tqdm import tqdm

from milvus_vector_database.src.milvus.milvus_glove import MilvusGlove
from milvus_vector_database.src.milvus.index_configuration import IndexConfig
from milvus_vector_database.scripts.load_glove_100_angular import load_glove_split


logger = logging.getLogger(__name__)


class BenchmarkResult:

    def __init__(self, index_config: IndexConfig, k: int):
        self.index_config = index_config
        self.k = k

        self.config_name = f"index={index_config.index_type.name}_k={k}"
        self.search_times_seconds: List[float] = []
        self.target_found: List[bool] = []

    def add_search_time(self, search_time: float) -> None:
        self.search_times_seconds.append(search_time)

    def add_accuracy_value(self, accuracy: bool) -> None:
        self.target_found.append(accuracy)

    def get_search_time_avg(self) -> float:
        return np.mean(self.search_times_seconds) if self.search_times_seconds else 0.0

    def get_search_time_std(self) -> float:
        return np.std(self.search_times_seconds) if len(self.search_times_seconds) > 1 else 0.0

    def get_accuracy(self) -> float:
        return np.mean(self.target_found) if self.target_found else 0.0

    def print_result(self) -> None:
        logger.info(f"Results for {self.config_name}. Avg search time: {self.get_search_time_avg()*1000}ms. Accuracy: {self.get_accuracy()*100:.2f}%.")


class MilvusGloveBenchmark:

    def __init__(self):
        self.milvus_client = MilvusGlove()
        self.test_data, self.neighbors_data = self.load_test_data()

    @staticmethod
    def load_test_data() -> Tuple[Dataset, Dataset]:
        """Load and prepare test data for benchmarking."""
        logger.info("Loading test dataset...")
        test = load_glove_split('test')
        neighbors = load_glove_split('neighbors')
        return test['test'], neighbors['neighbors']

    def warmup(self, n: int = 5) -> None:
        for i in range(n):
            self.milvus_client.search_vectors(query_vector=self.test_data[i: i+1]['emb'], k=1)

    def run_single_benchmark(self, index_config: IndexConfig, k: int) -> BenchmarkResult:
        """Benchmark a specific index configuration."""
        logger.info(f"Starting benchmark for {index_config}...")

        result = BenchmarkResult(index_config, k)

        # Warm up: run a few searches to stabilize performance
        self.warmup()

        # Benchmark search performance
        logger.info("Running search benchmark...")
        for i in range(len(self.test_data)):
            # Query closest embeddings
            start_time = time.time()
            res = self.milvus_client.search_vectors(query_vector=self.test_data[i: i+1]['emb'], k=k)
            search_time = time.time() - start_time
            
            # Get the target and the search results
            target = self.neighbors_data[i]['neighbors_id'][0]
            search_results = [r['id'] for r in res[0]]

            # Save the searchresult
            result.add_search_time(search_time)
            result.add_accuracy_value(target in search_results)

        logger.info(f"Completed benchmark for {index_config}.")
        return result

    def benchmark_single_index(self, preset_name, k_values):
        self.milvus_client.create_vector_index_from_preset(preset_name)
        results = []
        for k in k_values:
            result = self.run_single_benchmark(self.milvus_client.current_index_config, k)
            results.append(result)
        return results

    def benchmark_index_presets(self) -> Dict[str, BenchmarkResult]:
        """Run benchmark on all optimization presets."""
        presets = ["speed", "memory", "balanced", "accuracy", "scann", "gpu"]
        k_values = [2**i for i in range(6)]

        results = []
        pbar = tqdm(presets, desc="Benchmarking presets")
        for preset_name in pbar:
            pbar.set_description(f"Benchmarking preset '{preset_name}'")
            index_results = self.benchmark_single_index(preset_name, k_values)
            results.append(index_results)

        return results
