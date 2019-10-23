from unet.model.constants import *
from unet.model.preprocessing import input_fn, load_data
from unet.model.architecture import *
from unet.model.metrics import IOU
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import sys
import logging
import argparse

logformat = '%(asctime)s | %(levelname)s | [%(filename)s:%(lineno)s - %(funcName)s] %(message)s'
logging.basicConfig(format=logformat, level=logging.INFO, stream=sys.stdout)


parser = argparse.ArgumentParser()
parser.add_argument('--nr_epochs',
                    default=3,
                    type=int,
                    help="Number of passes over the entire data set for training.")
known_args, _ = parser.parse_known_args()
logging.info("Number of epochs: %s", known_args.nr_epochs)


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
    logging.info("Loading the data.")
    Xdata, ydata = load_data()  # expecting 4000 samples
    logging.info("Splitting the data into train and validation set.")
    train_size = 3200
    Xtrain, Xvalid, ytrain, yvalid = train_test_split(Xdata, ydata, train_size=train_size)
    logging.info("Creating train data input_fn.")
    train_batch_size = 32
    train_dataset = input_fn(Xtrain, ytrain, epochs=epochs, batch_size=train_batch_size, shuffle_buffer=300)
    logging.info("Creating validation data input_fn.")
    valid_batch_size = 200
    valid_dataset = input_fn(Xvalid, yvalid, epochs=None, batch_size=valid_batch_size, shuffle_buffer=None)

    logging.info("Creating keras callbacks.")
    checkpoint_path = "unet_saved_model/cp-{epoch:04d}.ckpt"
    # optional, add following parameters: monitor='mean_io_u', mode='max', save_best_only=True,
    model_checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)
    tensorboard = keras.callbacks.TensorBoard(log_dir='./logs')

    def scheduler(epoch):
        if epoch < 10:
            return 0.001
        else:
            return 0.001 * tf.math.exp(0.1 * (10 - epoch))

    learning_rate_decay = tf.keras.callbacks.LearningRateScheduler(scheduler)

    # Save the initialized weights using the `checkpoint_path` format
    model.save_weights(checkpoint_path.format(epoch=0))

    logging.info("Start training...")
    steps_per_epoch = train_size // train_batch_size
    model.fit(train_dataset,
              epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_data=valid_dataset,
              validation_steps=4,  # 4 steps of 200 samples covers the entire validation set
              callbacks=[model_checkpoint, tensorboard, learning_rate_decay])
    logging.info("Finished training!")


if __name__ == '__main__':
    model = get_model()
    train_and_validate(model, known_args.nr_epochs)
