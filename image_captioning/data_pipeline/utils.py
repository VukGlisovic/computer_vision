import os
from image_captioning.constants import DATA_DIR


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
