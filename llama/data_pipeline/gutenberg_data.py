import os
import re
from glob import glob
from tqdm import tqdm
import pandas as pd

from llama.constants import DATA_DIR


def how_to_get_gutenberg_subset():
    """Gives some simple instructions on how to get a subset
    of the gutenberg dataset.

    Returns:
        str: where the unzipped data should be stored.
    """
    folder_name = 'gutenberg_dataset_subset'
    save_path = os.path.join(DATA_DIR, folder_name)
    print("You can download the dataset from here:\n"
          "https://shibamoulilahiri.github.io/gutenberg_dataset.html\n"
          f"Put the unzipped folder in the data directory named '{folder_name}'\n"
          f"Thus in {save_path}.")
    return save_path


def preprocess_text(text):
    """Applies some simple preprocessing to the text.

    Args:
        text (str):

    Returns:
        str
    """
    # text = re.sub(r'(?<!\n)\n(?!\n)', '', text)  # remove single \n
    text = re.sub(r'\n\n+', '\n\n', text)
    return text


def load_gutenberg_data(data_dir):
    """Loads the gutenberg data from disk. Traverses the data_dir
    and looks for .txt files.

    Args:
        data_dir (str):

    Returns:
        tuple[dict, pd.DataFrame]
    """
    text_files = glob(os.path.join(data_dir, '**/*.txt'), recursive=True)
    text_dict = {}
    for filepath in tqdm(text_files, desc="Loading Gutenberg text data"):
        try:
            with open(filepath, 'r') as f:
                text_dict[os.path.basename(filepath)] = preprocess_text(f.read())
        except UnicodeDecodeError:
            # for simplicity, we'll ignore files we cannot load
            pass
    df_metadata = pd.DataFrame(data=[k.split('___') + [k] for k in text_dict.keys()], columns=['author', 'title', 'id'])
    return text_dict, df_metadata
