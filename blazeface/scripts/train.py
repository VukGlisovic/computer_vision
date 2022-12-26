import os
import sys
import logging
import argparse
from datetime import datetime

import tensorflow as tf

from blazeface.dataset import input_dataset
from blazeface.model import blazeface, losses

logformat = '%(asctime)s | %(levelname)s | [%(filename)s:%(lineno)s - %(funcName)s] %(message)s'
logging.basicConfig(format=logformat, level=logging.INFO, stream=sys.stdout)


def get_output_dir(experiment_dir):
    """Composes an output directory based on the current date and
    the run number of the day.

    Args:
        experiment_dir (str):

    Returns:
        str
    """
    date = datetime.now().strftime(format='%Y%m%d')
    run_nr = 0
    output_dir_template = os.path.join(experiment_dir, date + '_run{:03d}')
    output_dir = output_dir_template.format(run_nr)
    while os.path.isdir(output_dir):
        # increment run number until unused directory found
        run_nr += 1
        output_dir = output_dir_template.format(run_nr)
    logging.info("Output directory: %s", output_dir)
    return output_dir


def main(weights_path=None):
    """Combines all functionality
    """
    ds_train, _ = input_dataset.load_the300w_lp("train[:80%]")
    ds_train = input_dataset.create_input_dataset(ds_train, shuffle_buffer=512, batch_size=32, augment=True)
    logging.info("Loaded training dataset.")
    ds_validation, _ = input_dataset.load_the300w_lp("train[80%:]")
    ds_validation = input_dataset.create_input_dataset(ds_validation, batch_size=32)
    logging.info("Loaded validation dataset.")

    output_dir = get_output_dir('../data/experiments/')
    checkpoints_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(os.path.join(checkpoints_dir, 'weights-{epoch:02d}.hdf5'), monitor="val_loss", save_best_only=False, save_weights_only=True)
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(output_dir, 'tensorboard/'))
    learning_rate_cb = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_delta=0.0001, verbose=1),
    early_stop_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=20)
    callbacks = [checkpoint_cb, tensorboard_cb, learning_rate_cb, early_stop_cb]
    logging.info("Callback created.")

    model = blazeface.BlazeFaceModel()
    if weights_path:
        model.init_model_weights()
        logging.info("Loading pretrained weights from: %s", weights_path)
        model.load_weights(weights_path)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss={'deltas': losses.RegressionLoss(), 'labels': losses.ClassLoss()})
    logging.info("Created and compiled model.")
    model.fit(ds_train,
              validation_data=ds_validation,
              epochs=200,
              callbacks=callbacks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weights_path', type=str, required=False, help='Use this argument to start from some pretrained weights.')
    known_args, _ = parser.parse_known_args()
    main(known_args.weights_path)
