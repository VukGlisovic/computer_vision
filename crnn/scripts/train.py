import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from crnn.model.architecture import build_model
from crnn.model.ctc_loss import CTCLoss

logformat = '%(asctime)s | %(levelname)s | [%(filename)s:%(lineno)s - %(funcName)s] %(message)s'
logging.basicConfig(format=logformat, level=logging.INFO, stream=sys.stdout)

parser = argparse.ArgumentParser()


def main():
    model = build_model(10)
    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer,
                  loss=CTCLoss(blank_index=-1))
    model.summary()


if __name__ == '__main__':
    main()
