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
import matplotlib.pyplot as plt

from milvus_vector_database.constants import PROJECT_PATH
from milvus_vector_database.src.milvus.milvus_glove import MilvusGlove
from milvus_vector_database.src.milvus.index_configuration import IndexConfig
from milvus_vector_database.scripts.load_glove_100_angular import load_glove_split


logger = logging.getLogger(__name__)


class BenchmarkResult:

    def __init__(self, index_config: IndexConfig, k: int):
        self.index_config = index_config
        self.k = k

        self.config_name = f"index={index_config.index_type.name}__k={k}"
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

    def run_single_benchmark(self, index_config: IndexConfig, k: int) -> BenchmarkResult:
        """Benchmark a specific index configuration."""

        result = BenchmarkResult(index_config, k)

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

    def benchmark_single_index(self, preset_name: str, k_values: List[int]) -> None:
        self.milvus_client.create_vector_index_from_preset(preset_name)
        for k in k_values:
            result = self.run_single_benchmark(self.milvus_client.current_index_config, k)
            self.benchmark_results[preset_name].append(result)

    def benchmark_index_presets(self, preset_names: List[str], k_values: List[int]) -> None:
        """Run benchmark on all optimization presets."""

        pbar = tqdm(preset_names, desc="Benchmarking presets")
        for preset_name in pbar:
            pbar.set_description(f"Benchmarking preset '{preset_name}'")
            self.benchmark_single_index(preset_name, k_values)
        
        self.save_benchmark_results(os.path.join(PROJECT_PATH, 'data/benchmark_results.pkl'))
        self.plot_benchmark_results(os.path.join(PROJECT_PATH, 'data/benchmark_results.jpeg'))
    
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
    
    def plot_benchmark_results(self, output_path: str) -> None:
        """Plot the benchmark results."""
        fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(14, 8))
        fig.tight_layout(h_pad=5)

        line_kwargs = {'lw': 2, 'alpha': 0.7}
        for preset_name, preset_results in self.benchmark_results.items():
            xs = [result.k for result in preset_results]
            ys_time = [result.get_search_time_avg() * 1000 for result in preset_results]
            ys_acc = [result.get_accuracy() * 100 for result in preset_results]
            ax1.plot(xs, ys_time, label=preset_name, **line_kwargs)
            ax2.plot(xs, ys_acc, label=preset_name, **line_kwargs)

        ax1.legend()
        ax1.set_xlabel('k')
        ax1.set_ylabel('search time (ms)')
        ax1.set_ylim(0)
        ax1.set_title('Benchmark results (search time)')
        ax1.grid(ls='--', lw=0.5, c='black', alpha=0.4)
        ax1.set_xscale('log')
        ax1.set_xticks(xs)
        ax1.set_xticklabels([str(k) for k in xs])
        ax1.set_xticks([], minor=True)  # Remove minor ticks

        ax2.legend()
        ax2.set_xlabel('k')
        ax2.set_ylabel('accuracy (%)')
        ax2.set_ylim(0, 100)
        ax2.set_title('Benchmark results (accuracy)')
        ax2.grid(ls='--', lw=0.5, c='black', alpha=0.4)
        ax2.set_xscale('log')
        ax2.set_xticks(xs)
        ax2.set_xticklabels([str(k) for k in xs])
        ax2.set_xticks([], minor=True)  # Remove minor ticks

        if _dir := os.path.dirname(output_path):
            os.makedirs(_dir, exist_ok=True)
        plt.savefig(output_path)
        plt.close()
