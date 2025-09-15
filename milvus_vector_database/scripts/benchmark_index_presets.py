"""
Benchmark script to compare performance of different Milvus optimization strategies.

This script tests various index types and quantization methods to help you choose
the best optimization strategy for your specific use case.
"""
from typing import Dict, List
import os
import argparse
import logging

import matplotlib.pyplot as plt

from milvus_vector_database.src.benchmarking.benchmark_milvus_glove import MilvusGloveBenchmark
from milvus_vector_database.constants import PROJECT_PATH


logger = logging.getLogger(__name__)


def plot_preset_results(benchmark_results: Dict[str, List], output_path: str) -> None:
    """Plot the benchmark results."""
    fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(14, 8))
    fig.tight_layout(h_pad=5)

    line_kwargs = {'lw': 2, 'alpha': 0.7}
    for preset_name, preset_results in benchmark_results.items():
        xs = [result.k for result in preset_results]
        ys_time = [result.get_search_time_avg() * 1000 for result in preset_results]
        ys_acc = [result.get_accuracy() * 100 for result in preset_results]
        ax1.plot(xs, ys_time, label=preset_name, **line_kwargs)
        ax2.plot(xs, ys_acc, label=preset_name, **line_kwargs)

    ax1.legend()
    ax1.set_xlabel('k')
    ax1.set_ylabel('search time per query (ms)')
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


def main(output_filename: str) -> None:
    """Main function to run the benchmark."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

    presets = ["speed", "memory", "balanced", "accuracy", "scann", "gpu", "accuracy_gpu"]
    k_values = [2 ** i for i in range(6)]

    # Setup and run benchmark
    benchmark = MilvusGloveBenchmark()
    benchmark.benchmark_index_presets(presets, k_values)

    # Save benchmark results
    benchmark.save_benchmark_results(os.path.join(PROJECT_PATH, 'data', f'{output_filename}.pkl'))
    plot_preset_results(benchmark.benchmark_results, os.path.join(PROJECT_PATH, 'data', f'{output_filename}.jpeg'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark IVF_SQ8 index configurations")
    parser.add_argument("-o", "--output_filename", default='benchmark_result', help="Filename of the benchmark results without the extension. The extension will be added automatically.")
    args = parser.parse_args()

    main(args.output_filename)
