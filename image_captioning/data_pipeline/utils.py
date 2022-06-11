import os
import json
from image_captioning.constants import DATA_DIR


def load_json_file(path):
    """Reads a JSON file from disk into a python dictionary.

    Args:
        path (str):

    Returns:
        dict
    """
    with open(path, 'r') as f:
        dct = json.load(f)
    return dct


def imgpath_to_featurepath(path):
    """Converts a path pointing to an image a path pointing
    to its corresponding encoded features.

    Args:
        path (str):

    Returns:
        str
    """
    filename = os.path.basename(path)
    filename = filename.replace('.jpg', '.npy')
    return os.path.join(DATA_DIR, 'train2014_features', filename)  # store features in separate directory
