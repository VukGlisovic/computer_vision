import tensorflow as tf
import tensorflow_datasets as tfds


def load_the300w_lp(split="train[:80%]"):
    dataset, info = tfds.load("the300w_lp", split=split, with_info=True)
    return dataset, info
