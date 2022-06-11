import os
import json
import collections
from image_captioning.constants import *


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


def group_captions(annotations):
    """Group all captions together having the same image ID.

    Returns:
        tuple[dict, list]
    """
    imgpath_to_caption = collections.defaultdict(list)
    for ann in annotations['annotations']:
        caption = f"<start> {ann['caption']} <end>"  # add start and end token to sentences
        image_path = os.path.join(IMAGES_DIR, 'COCO_train2014_{:012d}.jpg'.format(ann['image_id']))
        imgpath_to_caption[image_path].append(caption)
    image_paths = list(imgpath_to_caption.keys())
    return imgpath_to_caption, image_paths
