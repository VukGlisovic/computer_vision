import os
import string
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

from efficientdet.model.efficientdet import efficientdet
from efficientdet.data_pipeline.input_dataset import create_combined_dataset
from efficientdet.model.losses import smooth_l1, focal


def main():
    model, model_prediction = efficientdet(phi=0)
    model.summary()

    ds_train = create_combined_dataset('../data/train.csv')
    ds_test = create_combined_dataset('../data/test.csv')

    adam = tf.keras.optimizers.Adam()
    losses = {'regression': smooth_l1(), 'classification': focal()}

    def scheduler(epoch, lr):
        return lr * tf.math.exp(-0.1)

    os.makedirs('../data/results/checkpoints/', exist_ok=True)
    callbacks = [
        tf.keras.callbacks.TensorBoard('../data/results/tensorboard', profile_batch=0),
        tf.keras.callbacks.ModelCheckpoint('../data/results/checkpoints/efficientdet-{epoch:02d}.hdf5', save_best_only=False, save_weights_only=True),
        tf.keras.callbacks.LearningRateScheduler(scheduler)
    ]

    model.compile(optimizer=adam, loss=losses)

    history = model.fit(ds_train, epochs=20, callbacks=callbacks)

    model.evaluate(ds_test)


if __name__ == '__main__':
    main()
