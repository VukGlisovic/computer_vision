"""
Benchmark script to compare performance of different configurations for the
IVF_SQ8 (Inverted File with Scalar Quantization 8-bit) optimization strategy.

This index type optimizes the database in two ways:
1. IVF: inverted file
   By organizing the vectors into clusters, it enables the search algorithm
   to focus only on the most relevant subsets of vectors.
2. SQ8: scalar quantization 8-bit
   The vectors are basically compressed from e.g. float32 (4 bytes) to int8
   (1 byte) which reduces the memory usage significantly.

There is much more detailed information here: https://milvus.io/docs/ivf-sq8.md

There's basically two types of parameters we can play around with:
1. Index build parameter: nlist
   Number of clusters to create using the k-means algorithm during index building.
2. Index search parameter: nprobe
   The number of clusters to search through. Generally the higher nprobe, the more
   accurate the search, but also the slower the search.

"""
import os
import argparse
import logging
from itertools import product

from milvus_vector_database.src.milvus.index_configuration import IndexConfig, IndexType, SearchConfig
from milvus_vector_database.src.benchmarking.benchmark_milvus_glove import MilvusGloveBenchmark
from milvus_vector_database.constants import PROJECT_PATH


logger = logging.getLogger(__name__)


def main(output_filename):
    """Main function to run the benchmark."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

    nlist = [128, 256, 512, 1024]
    nprobe = [8, 16]

    configs = [
        {
            'index_config': IndexConfig(index_type=IndexType.IVF_SQ8, metric_type="COSINE", params={"nlist": nl}),
            'search_config': SearchConfig(params={"nprobe": np}),
        }
        for nl, np in product(nlist, nprobe)  # First nlist then nprobe because then we don't need to re apply the same index config multiple times
    ]

    benchmark = MilvusGloveBenchmark()
    benchmark.benchmark_configs(configs, k=10)
    benchmark.save_benchmark_results(os.path.join(PROJECT_PATH, 'data', f'{output_filename}.pkl'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark IVF_SQ8 index configurations")
    parser.add_argument("-o", "--output_filename", default='benchmark_result', help="Filename of the benchmark results without the extension. The extension will be added automatically.")
    args = parser.parse_args()
    
    main(args.output_filename)
