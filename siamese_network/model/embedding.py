from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.applications import resnet
from siamese_network.constants import IMG_SPATIAL_SIZE


def create_embedding_model():
    """Creates an embedding model based on a ResNet50 model with imagenet weights.
    It makes only the last few layers trainable to make sure we only finetune
    the siamese network.

    Returns:
        tf.keras.models.Model
    """
    model_resnet = resnet.ResNet50(weights="imagenet", input_shape=IMG_SPATIAL_SIZE + (3,), include_top=False)

    flatten = layers.Flatten()(model_resnet.output)
    dense1 = layers.Dense(512, activation="relu")(flatten)
    dense1 = layers.BatchNormalization()(dense1)
    dense2 = layers.Dense(256, activation="relu")(dense1)
    dense2 = layers.BatchNormalization()(dense2)
    output = layers.Dense(256)(dense2)

    model_embedding = Model(model_resnet.input, output, name="Embedding")

    trainable = False
    for layer in model_resnet.layers:
        # loop over layers and make conv5_block1_out and everything after trainable
        if layer.name == "conv5_block1_out":
            trainable = True
        layer.trainable = trainable

    return model_embedding
