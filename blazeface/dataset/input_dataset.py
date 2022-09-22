import os
import tensorflow as tf
import tensorflow_datasets as tfds

from blazeface.constants import DATA_DIR


def load_the300w_lp(split="train[:80%]"):
    dataset, info = tfds.load("the300w_lp", split=split, data_dir=os.path.join(DATA_DIR, "tensorflow_datasets"), with_info=True)
    return dataset, info
