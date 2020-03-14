import os
import logging
from tensorflow import keras


def get_default_callbacks(monitor, mode, checkpoints_dir, tensorboard_logdir):
    """Specify what metric to monitor and a couple of standard callbacks
    will be created.

    Args:
        monitor (str):
        mode (str): whether to 'min' or 'max' the metric value
        checkpoints_dir (str):
        tensorboard_logdir (str):

    Returns:
        list
    """
    callbacks = []
    logging.info("Creating keras callbacks.")
    checkpoint_file_template = "cp-{epoch:04d}.ckpt"
    checkpoint_path = os.path.join(checkpoints_dir, checkpoint_file_template)
    # optional, add following parameters: monitor='mean_iou', mode='max', save_best_only=True,
    callbacks.append(keras.callbacks.ModelCheckpoint(checkpoint_path, monitor=monitor, mode=mode, save_weights_only=True, verbose=1))
    callbacks.append(keras.callbacks.TensorBoard(log_dir=tensorboard_logdir))
    callbacks.append(keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=0.5, patience=5, verbose=0, mode=mode, min_delta=0.0001, cooldown=0, min_lr=1e-5))
    callbacks.append(keras.callbacks.EarlyStopping(monitor=monitor, min_delta=0, patience=30, verbose=0, mode=mode, restore_best_weights=False))
    return callbacks
