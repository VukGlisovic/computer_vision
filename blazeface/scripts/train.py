import sys
import logging
import argparse
import tensorflow as tf

from blazeface.dataset import input_dataset
from blazeface.model import blazeface, losses

logformat = '%(asctime)s | %(levelname)s | [%(filename)s:%(lineno)s - %(funcName)s] %(message)s'
logging.basicConfig(format=logformat, level=logging.INFO, stream=sys.stdout)


def main():
    """Combines all functionality

    Args:
        config (dict):
    """
    ds_train, _ = input_dataset.load_the300w_lp("train[:80%]")
    ds_train = input_dataset.create_input_dataset(ds_train)
    logging.info("Loaded training dataset.")
    ds_validation, _ = input_dataset.load_the300w_lp("train[80%:]")
    ds_validation = input_dataset.create_input_dataset(ds_validation)
    logging.info("Loaded validation dataset.")

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint('../data/experiments/checkpoints/', monitor="val_loss", save_best_only=True, save_weights_only=True)
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir='../data/experiments/tensorboard/')
    learning_rate_cb = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1)
    callbacks = [checkpoint_cb, tensorboard_cb, learning_rate_cb]
    logging.info("Callback created.")

    model = blazeface.BlazeFaceModel()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss={'deltas': losses.RegressionLoss(), 'labels': losses.ClassLoss()})
                  # loss=[losses.RegressionLoss(), losses.ClassLoss()])
    logging.info("Created and compiled model.")
    model.fit(ds_train,
              validation_data=ds_validation,
              epochs=20,
              callbacks=callbacks)


if __name__ == '__main__':
    main()
