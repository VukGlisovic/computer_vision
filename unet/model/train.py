from unet.model.constants import *
from unet.model.preprocessing import input_fn, load_data
from unet.model.architecture import *
from unet.model.metrics import IOU
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import sys
import shutil
import logging
import argparse

logformat = '%(asctime)s | %(levelname)s | [%(filename)s:%(lineno)s - %(funcName)s] %(message)s'
logging.basicConfig(format=logformat, level=logging.INFO, stream=sys.stdout)


parser = argparse.ArgumentParser()
parser.add_argument('--nr_epochs',
                    default=10,
                    type=int,
                    help="Number of passes over the entire data set for training.")
parser.add_argument('--clean_up_previous_runs',
                    default=True,
                    type=bool,
                    help="Whether to remove old runs.")
known_args, _ = parser.parse_known_args()
logging.info("Number of epochs: %s", known_args.nr_epochs)


def get_train_validation_sets(epochs, train_batch_size=32):
    """Loads the data and creates a train/validation split.

    Args:
        epochs (int):
        train_batch_size (int):

    Returns:
        tuple[tf.dataset.Dataset, tf.dataset.Dataset]
    """
    Xdata, ydata = load_data()  # expecting 4000 samples
    logging.info("Splitting the data into train and validation set.")
    train_size = 3200
    Xtrain, Xvalid, ytrain, yvalid = train_test_split(Xdata, ydata, train_size=train_size)
    logging.info("Creating train data input_fn.")
    train_dataset = input_fn(Xtrain, ytrain, epochs=epochs, batch_size=train_batch_size, shuffle_buffer=300, augment=True)
    logging.info("Creating validation data input_fn.")
    valid_batch_size = 200
    valid_dataset = input_fn(Xvalid, yvalid, epochs=None, batch_size=valid_batch_size, shuffle_buffer=None, augment=False)
    return train_dataset, valid_dataset, train_size


def get_callbacks():
    """Instantiates the keras callbacks that can be used during training.

    Returns:
        tuple
    """
    # optional, add following parameters: monitor='mean_io_u', mode='max', save_best_only=True,
    model_checkpoint = keras.callbacks.ModelCheckpoint(CHECKPOINT_PATH, save_weights_only=True, verbose=1)
    tensorboard = keras.callbacks.TensorBoard(log_dir='./logs')

    def scheduler(epoch):
        if epoch < 10:
            return 0.001
        else:
            return 0.001 * tf.math.exp(0.1 * (10 - epoch))

    learning_rate_decay = tf.keras.callbacks.LearningRateScheduler(scheduler)
    return [model_checkpoint, tensorboard, learning_rate_decay]


def get_model():
    """Create and compile the model.

    Returns:
        keras.models.Model
    """
    unet_model = get_unet_model(batchnorm=False)

    optimizer = keras.optimizers.Adam(lr=0.01)
    unet_model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=[IOU(name='mean_io_u')])
    return unet_model


def train_and_validate(model, epochs):
    """Combines all methods and initiates the training.

    Args:
        model (keras.models.Model):
        epochs (int):
    """
    if known_args.clean_up_previous_runs:
        logging.info("Removing older runs.")
        shutil.rmtree('./logs', ignore_errors=True)
        shutil.rmtree(CHECKPOINT_DIR, ignore_errors=True)
    logging.info("Loading the data.")
    train_batch_size = 32
    train_dataset, valid_dataset, train_size = get_train_validation_sets(epochs, train_batch_size)

    logging.info("Creating keras callbacks.")
    callback_list = get_callbacks()

    # Save the initialized weights using the `checkpoint_path` format
    model.save_weights(CHECKPOINT_PATH.format(epoch=0))

    logging.info("Start training...")
    steps_per_epoch = (train_size // train_batch_size) // 2  # halve it to have more evaluation points
    model.fit(train_dataset,
              epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_data=valid_dataset,
              validation_steps=4,  # 4 steps of 200 samples covers the entire validation set
              callbacks=callback_list)
    logging.info("Finished training!")


if __name__ == '__main__':
    model = get_model()
    train_and_validate(model, known_args.nr_epochs)
