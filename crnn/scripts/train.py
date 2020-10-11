import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from crnn.data_pipeline.input_dataset import input_fn
from crnn.model.architecture import build_model
from crnn.model.ctc_loss import CTCLoss
from crnn.constants import *

logformat = '%(asctime)s | %(levelname)s | [%(filename)s:%(lineno)s - %(funcName)s] %(message)s'
logging.basicConfig(format=logformat, level=logging.INFO, stream=sys.stdout)

parser = argparse.ArgumentParser()


def create_dataset(path, epochs=None, batch_size=32, shuffle_buffer=None, augment=False):
    df = pd.read_csv(path, dtype=str)
    ds = input_fn(df, epochs, batch_size, shuffle_buffer, augment)
    return ds, len(df)


def get_model():
    model = build_model()
    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer,
                  loss=CTCLoss())
    model.summary()
    return model


def train():
    n_epochs = 10
    batch_size = 32
    ds_train, len_train = create_dataset(os.path.join(DIR_PROCESSED, 'train.csv'), epochs=n_epochs, batch_size=batch_size, shuffle_buffer=128, augment=True)
    ds_validation, len_validation = create_dataset(os.path.join(DIR_PROCESSED, 'validation.csv'), batch_size=128)
    ds_test, len_test = create_dataset(os.path.join(DIR_PROCESSED, 'test.csv'), batch_size=128)

    model = get_model()
    history = model.fit(
        ds_train,
        epochs=n_epochs,
        steps_per_epoch=int(np.ceil(len_train / batch_size)),
        # callbacks=callbacks,
        validation_data=ds_validation,
        validation_steps=int(np.ceil(len_validation / 128))
    )


if __name__ == '__main__':
    train()
