from unet.model.preprocessing import input_fn, load_data
from unet.model.architecture import *
from unet.model.metrics import IOU
from tensorflow import keras
from sklearn.model_selection import train_test_split
import sys
import shutil
import logging
import argparse
from distutils.dist import strtobool

logformat = '%(asctime)s | %(levelname)s | [%(filename)s:%(lineno)s - %(funcName)s] %(message)s'
logging.basicConfig(format=logformat, level=logging.INFO, stream=sys.stdout)


parser = argparse.ArgumentParser()
parser.add_argument('--nr_epochs',
                    default=3,
                    type=int,
                    help="Number of passes over the entire data set for training.")
parser.add_argument('--checkpoints_dir',
                    default='./unet_saved_model',
                    type=str,
                    help="Where to store the checkpoint files.")
parser.add_argument('--tensorboard_logdir',
                    default='./logs',
                    type=str,
                    help="Where to store the log files for Tensorboard.")
parser.add_argument('--remove_old_results',
                    default=True,
                    type=strtobool,
                    help="Whether to drop previous checkpoints and logs.")
known_args, _ = parser.parse_known_args()
logging.info("Number of epochs: %s", known_args.nr_epochs)
logging.info("Checkpoint dir: %s", known_args.checkpoints_dir)
logging.info("Tensorboard logs dir: %s", known_args.tensorboard_logdir)
logging.info("Remove old results: %s", known_args.remove_old_results)


def remove_old_results(checkpoints_dir, log_dir):
    """Deletes the directories in checkpoints_dir and log_dir if they
    exist. Otherwise does nothing.

    Args:
        checkpoints_dir (str):
        log_dir (str):
    """
    shutil.rmtree(checkpoints_dir, ignore_errors=True)
    shutil.rmtree(log_dir, ignore_errors=True)


def get_model():
    """Create and compile the model.

    Returns:
        keras.models.Model
    """
    unet_model = get_unet_model(batchnorm=False)

    optimizer = keras.optimizers.Adam(lr=0.01)
    unet_model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=[IOU(name='mean_io_u')])
    return unet_model


def train_and_validate(model, epochs, checkpoints_dir, log_dir):
    """Trains a Unet model on the TGS Salt Identification Challenge data set.
    First loads the data into a numpy array, creates a train/test split and
    creates tensorflow Dataset objects that can be fed to the fit method of
    the model.

    Args:
        model (keras.models.Model):
        epochs (int):
        checkpoints_dir (str):
        log_dir (str):
    """
    logging.info("Loading the data.")
    Xdata, ydata = load_data()  # expecting 4000 samples in a numpy array
    logging.info("Splitting the data into train and validation set.")
    train_size = 3200
    Xtrain, Xvalid, ytrain, yvalid = train_test_split(Xdata, ydata, train_size=train_size, random_state=42)
    logging.info("Creating train data input_fn.")
    train_batch_size = 32
    train_dataset = input_fn(Xtrain, ytrain, epochs=epochs, batch_size=train_batch_size, shuffle_buffer=300)
    logging.info("Creating validation data input_fn.")
    valid_batch_size = 200
    valid_dataset = input_fn(Xvalid, yvalid, epochs=None, batch_size=valid_batch_size, shuffle_buffer=None)

    logging.info("Creating keras callbacks.")
    checkpoint_file_template = "cp-{epoch:04d}.ckpt"
    checkpoint_path = os.path.join(checkpoints_dir, checkpoint_file_template)
    # optional, add following parameters: monitor='mean_io_u', mode='max', save_best_only=True,
    callback_model_checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss', save_weights_only=True, verbose=1)
    callback_tensorboard = keras.callbacks.TensorBoard(log_dir=log_dir)
    callback_reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=1e-8)

    # Save the initialized weights using the `checkpoint_path` format
    model.save_weights(checkpoint_path.format(epoch=0))

    logging.info("Start training...")
    steps_per_epoch = train_size // train_batch_size
    model.fit(train_dataset,
              epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_data=valid_dataset,
              validation_steps=4,  # 4 steps of 200 samples covers the entire validation set
              callbacks=[callback_model_checkpoint, callback_tensorboard, callback_reduce_lr])
    logging.info("Finished training!")


def main():
    """Combines all functionality into one function.
    """
    if known_args.remove_old_results:
        remove_old_results(known_args.checkpoints_dir, known_args.tensorboard_logdir)
    model = get_model()
    train_and_validate(model, known_args.nr_epochs, known_args.checkpoints_dir, known_args.tensorboard_logdir)


if __name__ == '__main__':
    main()
