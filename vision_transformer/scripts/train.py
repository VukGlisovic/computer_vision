import os
import re
import logging

import tensorflow as tf

from vision_transformer.constants import EXPERIMENTS_DIR
from vision_transformer.data_pipeline import input_dataset
from vision_transformer.model import model


def get_output_dir():
    experiment_dirs = os.listdir(EXPERIMENTS_DIR)
    nr_last_experiment = max([int(re.findall(r"run(\d+)", exp_dir)[0]) for exp_dir in experiment_dirs])
    output_dir = os.path.join(EXPERIMENTS_DIR, f'run{nr_last_experiment+1:03d}')
    os.makedirs(output_dir)
    return output_dir


def train():
    ds_train, ds_val, _ = input_dataset.get_cifar10_data_splits()
    for x, y in ds_train.take(1):
        img_shape = list(tf.shape(x).numpy())[1:]

    vit = model.build_ViT(img_shape)
    vit.summary()

    vit.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-3),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                         tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5 acc")])


    output_dir = get_output_dir()
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=os.path.join(output_dir, 'logs')),
        tf.keras.callbacks.ModelCheckpoint(os.path.join(output_dir, 'checkpoints/ckpt-{epoch:03d}.h5'), monitor='val_accuracy', save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.8, patience=4, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=1e-5, patience=10, verbose=0, mode="auto")
    ]

    history = vit.fit(ds_train,
                      validation_data=ds_val,
                      epochs=120,
                      callbacks=callbacks)


if __name__ == '__main__':
    train()
