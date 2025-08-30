"""
Benchmark script to compare performance of different Milvus optimization strategies.

This script tests various index types and quantization methods to help you choose
the best optimization strategy for your specific use case.
"""
import os
import time
import logging
import pickle
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
from datasets import Dataset
from tqdm import tqdm

from milvus_vector_database.src.milvus.milvus_glove import MilvusGlove
from milvus_vector_database.src.milvus.index_configuration import IndexConfig, SearchConfig, IndexOptimizationPresets
from milvus_vector_database.scripts.load_glove_100_angular import load_glove_split


logger = logging.getLogger(__name__)


class BenchmarkResult:

    def __init__(self, index_config: IndexConfig, search_config: SearchConfig, k: int):
        self.index_config = index_config
        self.search_config = search_config
        self.k = k

        self.config_name = f"{index_config}, {search_config}, k={k}"
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
        logger.info(f"Results for {self.config_name}. Avg search time: {self.get_search_time_avg()*1000:.2f}ms. Accuracy: {self.get_accuracy()*100:.2f}%.")


class MilvusGloveBenchmark:

    def __init__(self):
        self.milvus_client = MilvusGlove()
        self.test_data, self.neighbors_data = self.load_test_data()
        self.benchmark_results: Dict[str, List[BenchmarkResult]] = defaultdict(list)

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

    def benchmark_config(self, index_config: IndexConfig, search_config: SearchConfig = None, k: int = 1) -> BenchmarkResult:
        """Benchmark a specific index configuration."""

        result = BenchmarkResult(index_config, search_config, k)
        self.milvus_client.create_vector_index(index_config)
        self.milvus_client.set_search_config(search_config)

        # Warm up: run a few searches to stabilize performance
        self.warmup()

        # Benchmark search performance
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

        result.print_result()
        return result

    def benchmark_configs(self, configs: List[Dict], k=1):
        """configs should be a list of dicts containing 'index_config' and 'search_config'."""
        for config in configs:
            result = self.benchmark_config(index_config=config['index_config'], search_config=config['search_config'], k=k)
            self.benchmark_results[config['index_config'].index_type.name].append(result)

    def benchmark_index_presets(self, preset_names: List[str], k_values: List[int]) -> None:
        """Run benchmark on all optimization presets."""

        pbar = tqdm(preset_names, desc="Benchmarking presets")
        for preset_name in pbar:
            pbar.set_description(f"Benchmarking preset '{preset_name}'")
            index_config = IndexOptimizationPresets.get_preset(preset_name)
            for k in k_values:
                result = self.benchmark_config(index_config=index_config, k=k)
                self.benchmark_results[preset_name].append(result)
    
    def save_benchmark_results(self, output_path: str) -> None:
        """Save the benchmark results to a pickle file."""
        if _dir := os.path.dirname(output_path):
            os.makedirs(_dir, exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(self.benchmark_results, f)
        logger.info(f"Benchmark results saved to {output_path}")
    
    @staticmethod
    def load_benchmark_results(input_path: str) -> Dict[str, List[BenchmarkResult]]:
        """Load benchmark results from a pickle file."""
        with open(input_path, 'rb') as f:
            benchmark_results = pickle.load(f)
        return benchmark_results
