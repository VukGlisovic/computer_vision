import os
import sys
import logging
import argparse
import tensorflow as tf

from blazeface.dataset import input_dataset

logformat = '%(asctime)s | %(levelname)s | [%(filename)s:%(lineno)s - %(funcName)s] %(message)s'
logging.basicConfig(format=logformat, level=logging.INFO, stream=sys.stdout)


def main():
    """Combines all functionality

    Args:
        config (dict):
    """
    ds_train = input_dataset.load_the300w_lp("train[:80%]")
    logging.info("Loaded training dataset.")
    ds_validation = input_dataset.load_the300w_lp("train[80%:]")
    logging.info("Loaded validation dataset.")


if __name__ == '__main__':
    main()
