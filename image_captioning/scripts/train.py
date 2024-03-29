import os
import sys
import logging
import argparse
import tensorflow as tf
import matplotlib.pyplot as plt

from image_captioning.data_pipeline import input_dataset
from image_captioning.data_pipeline import utils
from image_captioning.model import text_vectorization, encoder, decoder, loss, checkpoint_manager
from image_captioning.constants import *

logformat = '%(asctime)s | %(levelname)s | [%(filename)s:%(lineno)s - %(funcName)s] %(message)s'
logging.basicConfig(format=logformat, level=logging.INFO, stream=sys.stdout)


embedding_dim = 256
units = 512
# Shape of the vector extracted from InceptionV3 is (64, 2048)
# These two variables represent that vector shape
features_shape = 2048
attention_features_shape = 64


def create_models(vocabulary_size):
    """Initializes all object that are necessary for training

    Args:
        vocabulary_size (int):

    Returns:
        tuple[tf.keras.models.Model, tf.keras.models.Model, tf.keras.optimizers.Optimizer]
    """
    logging.info("Creating encoder, decoder, optimizer and checkpoint manager.")
    cnn_encoder = encoder.CNN_Encoder(embedding_dim)
    rnn_decoder = decoder.RNN_Decoder(embedding_dim, units, vocabulary_size)
    optimizer = tf.keras.optimizers.Adam()
    return cnn_encoder, rnn_decoder, optimizer


@tf.function
def train_step(cnn_encoder, rnn_decoder, optimizer, word_to_index, feature_tensor, target):
    """Executes one training step.

    Args:
        cnn_encoder (tf.keras.models.Model):
        rnn_decoder (tf.keras.models.Model):
        optimizer (tf.keras.optimizers.Optimizer):
        word_to_index (tf.keras.layers.StringLookup):
        feature_tensor (tf.tensor):
        target (tf.tensor):

    Returns:
        tuple[tf.tensor, tf.tensor]
    """
    loss_value = 0

    # initializing the hidden state for each batch
    # because the captions are not related from image to image
    hidden = rnn_decoder.reset_state(batch_size=target.shape[0])

    dec_input = tf.expand_dims([word_to_index('<start>')] * target.shape[0], 1)

    with tf.GradientTape() as tape:
        features = cnn_encoder(feature_tensor)

        for i in range(1, target.shape[1]):
            # passing the features through the decoder
            predictions, hidden, _ = rnn_decoder(dec_input, features, hidden)

            loss_value += loss.loss_function(target[:, i], predictions)

            # using teacher forcing
            dec_input = tf.expand_dims(target[:, i], 1)

    total_loss = (loss_value / int(target.shape[1]))

    trainable_variables = cnn_encoder.trainable_variables + rnn_decoder.trainable_variables

    gradients = tape.gradient(loss_value, trainable_variables)

    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss_value, total_loss


def run_training(dataset, cnn_encoder, rnn_decoder, optimizer, word_to_index, ckpt_manager, epochs, num_steps_epoch):
    """Executes the full training.

    Args:
        dataset (tf.keras.data.Dataset):
        cnn_encoder (tf.keras.models.Model):
        rnn_decoder (tf.keras.models.Model):
        optimizer (tf.keras.optimizers.Optimizer):
        word_to_index (tf.keras.layers.StringLookup):
        ckpt_manager (tf.train.CheckpointManager):
        epochs (int):
        num_steps_epoch (int):

    Returns:
        list
    """
    # adding this in a separate cell because if you run the training cell many times, the loss_plot array will be reset
    loss_values = []

    logging.info("Start training...")
    for epoch in range(epochs):
        total_loss = 0

        for batch, (feature_tensor, target) in enumerate(dataset):
            batch_loss, t_loss = train_step(cnn_encoder, rnn_decoder, optimizer, word_to_index, feature_tensor, target)
            total_loss += t_loss

            if batch % 100 == 0:
                average_batch_loss = batch_loss.numpy() / int(target.shape[1])
                logging.info(f'Epoch {epoch} Batch {batch} Loss {average_batch_loss:.4f}')

        # storing the epoch end loss value to plot later
        loss_values.append(total_loss / num_steps_epoch)

        if epoch % 5 == 0:
            ckpt_manager.save()

        logging.info(f'Epoch {epoch} Loss {total_loss / num_steps_epoch:.6f}')
    logging.info("Finished training!")

    return loss_values


def plot_loss_curve(data_points, path):
    """Plots the resulting loss curve.

    Args:
        data_points (list):
        path (str): where to store the figure
    """
    fig, ax = plt.subplots(figsize=(15, 8))

    ax.set_title("Loss Curve", fontsize=20)
    ax.plot(data_points, lw=2.5, alpha=0.8)
    ax.grid(ls='--')
    ax.set_xlim(0)
    ax.set_ylim(0)
    ax.tick_params(labelsize=14)
    ax.set_xlabel("Epoch", fontsize=16)

    fig.savefig(path, format='png')
    plt.close()


def main(config):
    """Combines all functionality

    Args:
        config (dict):
    """
    epochs = config['epochs']
    batch_size = config['batch_size']
    buffer_size = config['buffer_size']

    all_captions, all_imgpaths, imgpath_to_caption = input_dataset.load_annotations()
    train_featurepaths, train_captions, val_featurepaths, val_captions = input_dataset.split_dataset(all_imgpaths, imgpath_to_caption)

    tokenizer = text_vectorization.load_text_vectorizer(TOKENIZER_PATH)
    train_dataset = input_dataset.create_input_dataset(train_featurepaths, train_captions, tokenizer, buffer_size, batch_size)

    cnn_encoder, rnn_decoder, optimizer = create_models(tokenizer.vocabulary_size())
    ckpt_manager = checkpoint_manager.create_checkpoint_manager(cnn_encoder, rnn_decoder, optimizer, restore_latest=True)

    # Create mappings for words to indices and indicies to words.
    word_to_index = tf.keras.layers.StringLookup(mask_token="", vocabulary=tokenizer.get_vocabulary())
    # index_to_word = tf.keras.layers.StringLookup(mask_token="", vocabulary=tokenizer.get_vocabulary(), invert=True)

    num_steps_per_epoch = len(train_featurepaths) // batch_size
    loss_values = run_training(train_dataset, cnn_encoder, rnn_decoder, optimizer, word_to_index, ckpt_manager, epochs, num_steps_per_epoch)

    plot_loss_curve(loss_values, os.path.join(EXPERIMENT_DIR, 'loss_curve.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config',
                        default='../config.json',
                        help='Path to configuration file')
    known_args, _ = parser.parse_known_args()
    config_dict = utils.load_json_file(known_args.config)

    main(config_dict)
