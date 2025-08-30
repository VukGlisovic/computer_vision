"""
Benchmark script to compare performance of different Milvus optimization strategies.

This script tests various index types and quantization methods to help you choose
the best optimization strategy for your specific use case.
"""
import os
import argparse
import logging

from milvus_vector_database.src.benchmarking.benchmark_milvus_glove import MilvusGloveBenchmark
from milvus_vector_database.constants import PROJECT_PATH


logger = logging.getLogger(__name__)


def main(output_filename: str):
    """Main function to run the benchmark."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

    presets = ["speed", "memory", "balanced", "accuracy", "scann", "gpu", "accuracy_gpu"]
    k_values = [2 ** i for i in range(6)]

    benchmark = MilvusGloveBenchmark()
    benchmark.benchmark_index_presets(presets, k_values)
    benchmark.save_benchmark_results(os.path.join(PROJECT_PATH, 'data', f'{output_filename}.pkl'))
    benchmark.plot_benchmark_results(os.path.join(PROJECT_PATH, 'data', f'{output_filename}.jpeg'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark IVF_SQ8 index configurations")
    parser.add_argument("output_filename", default='benchmark_result', help="Filename of the benchmark results without the extension. The extension will be added automatically.")
    args = parser.parse_args()

    main(args.output_filename)
