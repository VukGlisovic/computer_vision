import os
import logging
import tensorflow as tf
from glob import glob

from siamese_network.data_pipeline import input_dataset
from siamese_network.model.siamese_model import SiameseModel


def create_train_validation_tf_dataset():
    """Creates training and validation tf datasets.

    Returns:
        tuple[tf.data.Dataset, tf.data.Dataset]
    """
    anchor_images = glob("../data/left/*")
    positive_images = glob("../data/right/*")
    logging.info("Number of anchor / positive images: %s / %s", len(anchor_images), len(positive_images))
    ds_train, ds_validation = input_dataset.create_triplet_dataset(anchor_images, positive_images, batch_size=32)
    return ds_train, ds_validation


def main():
    """Executes all necessary functions to start a training. This means:
    building the model, loading the data, preparing callbacks and starting
    the training.
    """

    ds_train, ds_validation = create_train_validation_tf_dataset()

    output_dir = '../data/results/'
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_template_path = os.path.join(checkpoint_dir, 'siamese-{epoch:02d}.hdf5')
    callbacks = [
        tf.keras.callbacks.TensorBoard(os.path.join(output_dir, 'tensorboard'), profile_batch=0),
        tf.keras.callbacks.ModelCheckpoint(checkpoint_template_path, monitor='val_loss', save_best_only=True, save_weights_only=True)
    ]

    model = SiameseModel(margin=0.5)
    model(next(ds_train.take(1).as_numpy_iterator()))  # call the model for a summary
    model.summary()

    adam = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=adam)

    history = model.fit(ds_train, validation_data=ds_validation, epochs=25, callbacks=callbacks)


if __name__ == '__main__':
    main()
