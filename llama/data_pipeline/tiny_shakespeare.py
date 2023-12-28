import os
import urllib

from llama.constants import DATA_DIR


def download_tiny_shakespeare(file_name="tinyshakespeare.txt"):
    """Downloads tiny shakespeare dataset and saves it to disk. Returns the
    path to the saved file.
    """
    # the URL of the raw text file on GitHub
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    # store in DATA_DIR
    os.makedirs(DATA_DIR, exist_ok=True)
    save_path = os.path.join(DATA_DIR, file_name)
    if os.path.isfile(save_path):
        print(f"Tiny shakespeare already downloaded to: '{save_path}'")
    else:
        urllib.request.urlretrieve(url, save_path)
        print(f"Downloaded tiny shakespeare to: '{save_path}'")
    return save_path
