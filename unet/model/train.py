from unet.model.preprocessing import input_fn, load_data
from unet.model.architecture import *
from unet.model.metrics import IOU
from tensorflow import keras
from sklearn.model_selection import train_test_split
import sys
import shutil
import logging
import argparse
import json
from distutils.dist import strtobool

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
    unet_model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=[IOU(name='mean_io_u')])
    return unet_model


def train_and_validate(model, nr_epochs, batch_size, shuffle_buffer, checkpoints_dir, tensorboard_logdir):
    """Trains a Unet model on the TGS Salt Identification Challenge data set.
    First loads the data into a numpy array, creates a train/test split and
    creates tensorflow Dataset objects that can be fed to the fit method of
    the model.

    Args:
        model (keras.models.Model):
        nr_epochs (int):
        batch_size (int):
        shuffle_buffer (int):
        checkpoints_dir (str):
        tensorboard_logdir (str):
    """
    logging.info("Loading the data.")
    Xdata, ydata = load_data()  # expecting 4000 samples in a numpy array
    logging.info("Splitting the data into train and validation set.")
    train_size = 3200
    Xtrain, Xvalid, ytrain, yvalid = train_test_split(Xdata, ydata, train_size=train_size, random_state=42)
    logging.info("Creating train data input_fn.")
    train_dataset = input_fn(Xtrain, ytrain, epochs=nr_epochs, batch_size=batch_size, shuffle_buffer=shuffle_buffer)
    logging.info("Creating validation data input_fn.")
    valid_dataset = input_fn(Xvalid, yvalid, epochs=None, batch_size=200, shuffle_buffer=None)

    logging.info("Creating keras callbacks.")
    checkpoint_file_template = "cp-{epoch:04d}.ckpt"
    checkpoint_path = os.path.join(checkpoints_dir, checkpoint_file_template)
    # optional, add following parameters: monitor='mean_io_u', mode='max', save_best_only=True,
    callback_model_checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss', save_weights_only=True, verbose=1)
    callback_tensorboard = keras.callbacks.TensorBoard(log_dir=tensorboard_logdir)
    callback_reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=1e-8)

    # Save the initialized weights using the `checkpoint_path` format
    model.save_weights(checkpoint_path.format(epoch=0))

    logging.info("Start training...")
    steps_per_epoch = train_size // batch_size
    model.fit(train_dataset,
              epochs=nr_epochs,
              steps_per_epoch=steps_per_epoch,
              validation_data=valid_dataset,
              validation_steps=4,  # 4 steps of 200 samples covers the entire validation set
              callbacks=[callback_model_checkpoint, callback_tensorboard, callback_reduce_lr])
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
