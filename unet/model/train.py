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

from unet.model.preprocessing import create_data_generators
from unet.model.architecture import *
from unet.model.callbacks import get_default_callbacks

logformat = '%(asctime)s | %(levelname)s | [%(filename)s:%(lineno)s - %(funcName)s] %(message)s'
logging.basicConfig(format=logformat, level=logging.INFO, stream=sys.stdout)


parser = argparse.ArgumentParser()
parser.add_argument('--configuration_file',
                    default='configs/binary_cross_entropy.json',
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
for k,v in config.items():
    logging.info("%s: %s", k, v)


def remove_old_results(checkpoints_dir, tensorboard_logdir, **kwargs):
    """Deletes the directories in checkpoints_dir and tensorboard_logdir if they
    exist. Otherwise does nothing.

    Args:
        checkpoints_dir (str):
        tensorboard_logdir (str):
    """
    shutil.rmtree(checkpoints_dir, ignore_errors=True)
    shutil.rmtree(tensorboard_logdir, ignore_errors=True)


def train_and_validate(model, metric, nr_epochs, batch_size, shuffle_buffer, checkpoints_dir, tensorboard_logdir, **kwargs):
    """Trains a Unet model on the TGS Salt Identification Challenge data set.
    First loads the data into a numpy array, creates a train/test split and
    creates tensorflow Dataset objects that can be fed to the fit method of
    the model.

    Args:
        model (keras.models.Model): the Unet model.
        metric (str): the metric to optimize during training.
        nr_epochs (int): the number of times to iterate through the entire
            training data set.
        batch_size (int): the number of samples in a mini-batch.
        shuffle_buffer (int): number of samples to shuffle in a shuffle buffer.
        checkpoints_dir (str): where to store the checkpoint files.
        tensorboard_logdir (str): where to store the Tensorboard logs.
    """
    train_dataset, valid_dataset = create_data_generators(nr_epochs, batch_size, shuffle_buffer)

    callback_list = get_default_callbacks('val_{}'.format(metric), checkpoints_dir, tensorboard_logdir)

    logging.info("Start training...")
    steps_per_epoch = 3200 // batch_size
    model.fit(train_dataset,
              epochs=nr_epochs,
              steps_per_epoch=steps_per_epoch,
              validation_data=valid_dataset,
              validation_steps=4,  # 4 steps of 200 samples covers the entire validation set
              callbacks=callback_list)
    logging.info("Finished training!")


def main():
    """Combines all functionality into one function.
    """
    if known_args.remove_old_results:
        remove_old_results(**config)
    model = get_unet_model(**config)
    train_and_validate(model, **config)


if __name__ == '__main__':
    main()
