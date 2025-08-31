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
from typing import Dict, List
import os
import argparse
import logging
from itertools import product

import numpy as np
import matplotlib.pyplot as plt

from milvus_vector_database.src.milvus.index_configuration import IndexConfig, IndexType, SearchConfig
from milvus_vector_database.src.benchmarking.benchmark_milvus_glove import MilvusGloveBenchmark
from milvus_vector_database.constants import PROJECT_PATH


logger = logging.getLogger(__name__)


def plot_ivf_sq8_results(benchmark_results: Dict[str, List], output_path: str):
    """Plot the IVF_SQ8 benchmark results.

    The x-axis will be based on the `nlist` parameter from the index config and the
    different lines will be based on the `nprobe` from the search config.
    """
    fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(14, 8))
    fig.tight_layout(h_pad=5)

    unique_nprobes = np.unique([r.search_config.params['nprobe'] for r in benchmark_results['IVF_SQ8']])

    line_kwargs = {'lw': 2, 'alpha': 0.7}
    for nprobe in unique_nprobes:

        nprobe_results = [r for r in benchmark_results['IVF_SQ8'] if r.search_config.params['nprobe'] == nprobe]
        xs = [r.index_config.params['nlist'] for r in nprobe_results]  # Only needed once, but for simplicity repeating it `nprobe` times
        ys_time = [r.get_search_time_avg() * 1000 for r in nprobe_results]
        ys_acc = [r.get_accuracy() * 100 for r in nprobe_results]

        ax1.plot(xs, ys_time, label=f'nprobe={nprobe}', **line_kwargs)
        ax2.plot(xs, ys_acc, label=f'nprobe={nprobe}', **line_kwargs)

    ax1.legend()
    ax1.set_xlabel('nlist')
    ax1.set_ylabel('search time per query (ms)')
    ax1.set_ylim(0)
    ax1.set_title('Benchmark results (search time)')
    ax1.grid(ls='--', lw=0.5, c='black', alpha=0.4)
    ax1.set_xscale('log')
    ax1.set_xticks(xs)
    ax1.set_xticklabels([str(k) for k in xs])
    ax1.set_xticks([], minor=True)  # Remove minor ticks

    ax2.legend()
    ax2.set_xlabel('nlist')
    ax2.set_ylabel('accuracy (%)')
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


def main(output_filename):
    """Main function to run the benchmark."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

    nlist = [128, 256, 512, 1024, 2048]
    nprobe = [1, 2, 4, 8, 16, 32]

    configs = [
        {
            'index_config': IndexConfig(index_type=IndexType.IVF_SQ8, metric_type="COSINE", params={"nlist": nl}),
            'search_config': SearchConfig(params={"nprobe": np}),
        }
        for nl, np in product(nlist, nprobe)  # First nlist then nprobe because then we don't need to re apply the same index config multiple times
    ]

    # Setup and run benchmark
    benchmark = MilvusGloveBenchmark()
    benchmark.benchmark_configs(configs, k=10)

    # Save benchmark results
    benchmark.save_benchmark_results(os.path.join(PROJECT_PATH, 'data', f'{output_filename}.pkl'))
    plot_ivf_sq8_results(benchmark.benchmark_results, os.path.join(PROJECT_PATH, 'data', f'{output_filename}.jpeg'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark IVF_SQ8 index configurations")
    parser.add_argument("-o", "--output_filename", default='benchmark_result', help="Filename of the benchmark results without the extension. The extension will be added automatically.")
    args = parser.parse_args()
    
    main(args.output_filename)
