"""
Script to download and load the glove-100-angular dataset from Hugging Face.

This script provides multiple methods to download the GloVe 100-dimensional 
angular dataset from Hugging Face for use with vector databases like Milvus.
"""
from typing import Tuple
import os
import logging

import numpy as np
from datasets import load_dataset, DatasetDict, Dataset

from milvus_vector_database.src.milvus.milvus_glove import MilvusGlove
from milvus_vector_database.constants import PROJECT_PATH

logger = logging.getLogger(__name__)


def load_glove_split(split_name: str, force_download: bool = False) -> DatasetDict:
    """
    Load a specific split from the glove-100-angular dataset.
    """
    download_mode = "force_redownload" if force_download else "reuse_cache_if_exists"
    cache_dir = os.path.join(PROJECT_PATH, 'data/.hf')

    dataset_split = load_dataset(
        "open-vdb/glove-100-angular",
        name=split_name,
        cache_dir=cache_dir,
        download_mode=download_mode
    )
    return dataset_split


def download_glove_dataset(force_download: bool = False) -> DatasetDict:
    """
    Download the glove-100-angular dataset from Hugging Face.
    
    Args:
        force_download (bool): If True, re-download even if cached.
        
    Returns:
        dict: Dictionary containing the loaded dataset splits
    """
    logger.info("Downloading glove-100-angular dataset from Hugging Face...")

    # Available configs for this dataset
    configs = ['train', 'test', 'neighbors']
    dataset = DatasetDict()  # We'll update this DatasetDict with all the dataset splits
    
    # We need to load each config separately
    for config in configs:
        dataset_split = load_glove_split(config, force_download)
        dataset.update(dataset_split)

    logger.info(f"Dataset downloaded successfully!")
    return dataset


def get_embeddings_and_ids(dataset_split: Dataset) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract embeddings and IDs from a dataset split.
    
    Args:
        dataset_split: A single split from the dataset (e.g., dataset['train'])
        
    Returns:
        tuple: (embeddings_array, ids_array) as numpy arrays
    """
    # Convert to numpy arrays for easier manipulation
    logger.info(f"Extracting embeddings and IDs from dataset split and converting to numpy arrays.")
    embeddings = np.array(dataset_split['emb'])
    ids = np.array(dataset_split['idx'])
    
    logger.info(f"Extracted {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
    return embeddings, ids


def main():
    """
    Main function to demonstrate dataset downloading and processing.
    """
    # Configure logging to show INFO level messages in standard output
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]  # This ensures output goes to stdout
    )
    
    # Download the dataset
    dataset = download_glove_dataset()

    milvus_glove = MilvusGlove()
    milvus_glove.create_collection(overwrite=True)
    
    # Process each split
    for split_name in dataset.keys():
        logger.info(f"Processing {split_name} split...")
        
        # Get the actual dataset split (each config returns a DatasetDict)
        split_data = dataset[split_name]
        
        # For train and test splits, extract embeddings
        if split_name in ['train', 'test']:

            embeddings, ids = get_embeddings_and_ids(split_data)

            if split_name == 'train':
                # We only want to insert the train split into the database
                milvus_glove.insert_vectors(embeddings, ids)

            print(f"{split_name} split summary:")
            print(f"  - Number of vectors: {len(embeddings)}")
            print(f"  - Vector dimension: {embeddings.shape[1]}")
            print(f"  - Data type: {embeddings.dtype}")
            print(f"  - ID range: {ids.min()} to {ids.max()}")
        
        # For neighbors split, show structure
        elif split_name == 'neighbors':
            print(f"Neighbors split contains ground truth nearest neighbors:")
            print(f"  - Features: {list(split_data.features.keys())}")
            print(f"  - Number of queries: {len(split_data)}")
            if len(split_data) > 0:
                first_item = split_data[0]
                print(f"  - Example query ID: {first_item['idx']}")
                print(f"  - Number of neighbors per query: {len(first_item['neighbors_id'])}")
                print(f"  - Distance metric: {first_item['metric']}")


if __name__ == "__main__":
    main()
