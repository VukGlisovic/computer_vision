"""
In order to train a Unet model with this script, you need to provide
a configuration json file. This file should contain all parameters
that are required by the train_and_validate method except for 'model'.
"""
import sys
import logging
import argparse
import shutil
import json
from distutils.dist import strtobool

from tensorflow import keras

from unet.model.preprocessing import create_data_generators
from unet.model.architecture import *
from unet.model.metrics import iou_thr_05

logformat = '%(asctime)s | %(levelname)s | [%(filename)s:%(lineno)s - %(funcName)s] %(message)s'
logging.basicConfig(format=logformat, level=logging.INFO, stream=sys.stdout)


parser = argparse.ArgumentParser()
parser.add_argument('--configuration_file',
                    default='./config.json',
                    type=str,
                    help="JSON file containing all the configurations required for training. For instance "
                         "number of epochs, batch size, etc.")
parser.add_argument('--remove_old_results',
                    default=True,
                    type=strtobool,
                    help="Whether to drop previous checkpoints and logs.")
known_args, _ = parser.parse_known_args()
logging.info("Configuration file: %s", known_args.configuration_file)
logging.info("Remove old results: %s", known_args.remove_old_results)

with open(known_args.configuration_file, 'r') as f:
    config = json.load(f)
logging.info("Number of epochs: %s", config['nr_epochs'])
logging.info("Batch size: %s", config['batch_size'])
logging.info("Checkpoint dir: %s", config['checkpoints_dir'])
logging.info("Tensorboard logs dir: %s", config['tensorboard_logdir'])


def remove_old_results(checkpoints_dir, tensorboard_logdir, **kwargs):
    """Deletes the directories in checkpoints_dir and tensorboard_logdir if they
    exist. Otherwise does nothing.

    Args:
        checkpoints_dir (str):
        tensorboard_logdir (str):
    """
    shutil.rmtree(checkpoints_dir, ignore_errors=True)
    shutil.rmtree(tensorboard_logdir, ignore_errors=True)


def get_model():
    """Create and compile the model.

    Returns:
        keras.models.Model
    """
    unet_model = get_unet_model(batchnorm=False)

    optimizer = keras.optimizers.Adam(lr=0.01)
    unet_model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=[iou_thr_05])
    return unet_model


def train_and_validate(model, nr_epochs, batch_size, shuffle_buffer, checkpoints_dir, tensorboard_logdir):
    """Trains a Unet model on the TGS Salt Identification Challenge data set.
    First loads the data into a numpy array, creates a train/test split and
    creates tensorflow Dataset objects that can be fed to the fit method of
    the model.

    Args:
        model (keras.models.Model): the Unet model.
        nr_epochs (int): the number of times to iterate through the entire
            training data set.
        batch_size (int): the number of samples in a mini-batch.
        shuffle_buffer (int): number of samples to shuffle in a shuffle buffer.
        checkpoints_dir (str): where to store the checkpoint files.
        tensorboard_logdir (str): where to store the Tensorboard logs.
    """
    train_dataset, valid_dataset = create_data_generators(nr_epochs, batch_size, shuffle_buffer)

    logging.info("Creating keras callbacks.")
    checkpoint_file_template = "cp-{epoch:04d}.ckpt"
    checkpoint_path = os.path.join(checkpoints_dir, checkpoint_file_template)
    # optional, add following parameters: monitor='mean_iou', mode='max', save_best_only=True,
    monitor = 'val_iou_thr_05'
    callback_model_checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_path, monitor=monitor, save_weights_only=True, verbose=1)
    callback_tensorboard = keras.callbacks.TensorBoard(log_dir=tensorboard_logdir)
    callback_reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=0.5, patience=5, verbose=0, mode='min', min_delta=0.0001, cooldown=0, min_lr=1e-6)
    callback_early_stop = keras.callbacks.EarlyStopping(monitor=monitor, min_delta=0, patience=15, verbose=0, mode='min', restore_best_weights=False)

    # Save the initialized weights using the `checkpoint_path` format
    model.save_weights(checkpoint_path.format(epoch=0))

    logging.info("Start training...")
    steps_per_epoch = 3200 // batch_size
    model.fit(train_dataset,
              epochs=nr_epochs,
              steps_per_epoch=steps_per_epoch,
              validation_data=valid_dataset,
              validation_steps=4,  # 4 steps of 200 samples covers the entire validation set
              callbacks=[callback_model_checkpoint, callback_tensorboard, callback_reduce_lr, callback_early_stop])
    logging.info("Finished training!")


def main():
    """Combines all functionality into one function.
    """
    if known_args.remove_old_results:
        remove_old_results(**config)
    model = get_model()
    train_and_validate(model, **config)


if __name__ == '__main__':
    main()
