import os
import re
import argparse

import yaml
import tensorflow as tf

from vision_transformer.constants import EXPERIMENTS_DIR
from vision_transformer.data_pipeline import input_dataset
from vision_transformer.model import model


def load_config(config_path):
    """
    Args:
        config_path (str):

    Returns:
        dict
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def get_output_dir():
    """ Dynamically determines the new output directory. """
    experiment_dirs = os.listdir(EXPERIMENTS_DIR)
    nr_last_experiment = max([int(re.findall(r"run(\d+)", exp_dir)[0]) for exp_dir in experiment_dirs])
    output_dir = os.path.join(EXPERIMENTS_DIR, f'run{nr_last_experiment+1:03d}')
    os.makedirs(output_dir)
    return output_dir


def train(config):
    """ Creates dataloaders, builds the model and runs a training. """
    ds_train, ds_val, _ = input_dataset.get_cifar10_data_splits(config['train']['batch_size'])
    for x, y in ds_train.take(1):
        img_shape = list(tf.shape(x).numpy())[1:]

    vit = model.build_ViT(img_shape, config['model'])
    vit.summary()

    vit.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config['train']['learning_rate']),
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
                      epochs=config['train']['epochs'],
                      callbacks=callbacks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_path', default='config.yaml', type=str, help='Path to yaml file.')
    known_args, _ = parser.parse_known_args()

    train(
        load_config(known_args.config_path)
    )
