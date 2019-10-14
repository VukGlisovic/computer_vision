from unet.model.constants import *
from unet.model.preprocessing import input_fn, load_data
from unet.model.architecture import *
from unet.model.metrics import iou
from tensorflow import keras
from sklearn.model_selection import train_test_split
import sys
import logging

logformat = '%(asctime)s | %(levelname)s | [%(filename)s:%(lineno)s - %(funcName)s] %(message)s'
logging.basicConfig(format=logformat, level=logging.INFO, stream=sys.stdout)


def get_model():
    input_image = keras.layers.Input(IMAGE_SHAPE, name='image')
    unet_model = create_unet_model(input_image, batchnorm=False)

    optimizer = keras.optimizers.Adam(lr=0.01)
    unet_model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=[iou])
    return unet_model


def train_and_validate(model):
    logging.info("Loading the data.")
    Xdata, ydata = load_data()  # expecting 4000 samples
    logging.info("Splitting the data into train and validation set.")
    train_size = 3200
    Xtrain, Xvalid, ytrain, yvalid = train_test_split(Xdata, ydata, train_size=train_size)
    logging.info("Creating train data input_fn.")
    train_batch_size = 32
    train_dataset = input_fn(Xtrain, ytrain, epochs=2, batch_size=train_batch_size, shuffle_buffer=300)
    logging.info("Creating validation data input_fn.")
    valid_batch_size = 200
    valid_dataset = input_fn(Xvalid, yvalid, epochs=None, batch_size=valid_batch_size, shuffle_buffer=None)

    logging.info("Creating keras callbacks.")
    model_checkpoint = keras.callbacks.ModelCheckpoint('unet_saved_model', monitor='iou', mode='max', save_best_only=True, verbose=1)

    logging.info("Start training...")
    steps_per_epoch = train_size // train_batch_size
    model.fit(train_dataset,
              steps_per_epoch=steps_per_epoch,
              validation_data=valid_dataset,
              validation_steps=4,  # 4 steps of 200 samples covers the entire validation set
              callbacks=[model_checkpoint],
              verbose=2)
    logging.info("Finished training!")


if __name__ == '__main__':
    model = get_model()
    train_and_validate(model)
