import logging
import tensorflow as tf

from image_captioning.constants import CHECKPOINT_PATH


def create_checkpoint_manager(cnn_encoder, rnn_decoder, optimizer, restore_latest=False):
    """Creates a checkpoint manager for writing and loading of checkpoints.

    Args:
        cnn_encoder (tf.keras.models.Model):
        rnn_decoder (tf.keras.models.Model):
        optimizer (tf.keras.optimizer.Optimizer):
        restore_latest (bool):

    Returns:
        tf.train.Checkpoint
    """
    ckpt = tf.train.Checkpoint(encoder=cnn_encoder,
                               decoder=rnn_decoder,
                               optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, CHECKPOINT_PATH, max_to_keep=5)

    if restore_latest and ckpt_manager.latest_checkpoint:
        logging.info("Restoring the latest checkpoint in checkpoint_path.")
        ckpt.restore(ckpt_manager.latest_checkpoint)

    return ckpt_manager
