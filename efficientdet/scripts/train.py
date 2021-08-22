import os
import tensorflow as tf

from efficientdet.model.efficientdet import EfficientDet
from efficientdet.data_pipeline.input_dataset import create_combined_dataset
from efficientdet.model.losses import HuberRegressionLoss, FocalClassificationLoss


def main():
    """Executes all necessary functions to start a training. This means:
    building the model, loading the data, preparing callbacks and starting
    the training.
    """
    model = EfficientDet(phi=0)
    model.build((None, None, None, 1))
    model.decode_outputs = False
    model.summary()

    ds_train = create_combined_dataset('../data/train.csv')
    ds_test = create_combined_dataset('../data/test.csv')

    def scheduler(epoch, lr):
        return lr * tf.math.exp(-0.1)

    output_dir = '../data/results/'
    os.makedirs(os.path.join(output_dir, 'checkpoints/'), exist_ok=True)
    callbacks = [
        tf.keras.callbacks.TensorBoard(os.path.join(output_dir, 'tensorboard'), profile_batch=0),
        tf.keras.callbacks.ModelCheckpoint(os.path.join(output_dir, 'checkpoints/efficientdet-{epoch:02d}.hdf5'), save_best_only=False, save_weights_only=True),
        tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)
    ]

    # set learning rate to 0.001 / exp(-0.1)
    adam = tf.keras.optimizers.Adam(learning_rate=0.001105)
    losses = {'regression': HuberRegressionLoss(), 'classification': FocalClassificationLoss(num_classes=10)}

    model.compile(optimizer=adam, loss=losses)

    history = model.fit(ds_train, validation_data=ds_test, epochs=30, callbacks=callbacks)


if __name__ == '__main__':
    main()
