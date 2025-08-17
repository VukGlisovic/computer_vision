"""
Benchmark script to compare performance of different Milvus optimization strategies.

This script tests various index types and quantization methods to help you choose
the best optimization strategy for your specific use case.
"""
import logging

from milvus_vector_database.src.benchmarking.benchmark_milvus_glove import MilvusGloveBenchmark

logger = logging.getLogger(__name__)


def main():
    """Main function to run the benchmark."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

    benchmark = MilvusGloveBenchmark()

    benchmark.benchmark_index_presets()


if __name__ == "__main__":
    main()
