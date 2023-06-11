import tensorflow as tf

from vision_transformer.constants import CIFAR10_CLASSES
from vision_transformer.model import model_builder_functions


# hyperparameters
transformer_layers = 6
patch_size = 4  # nr pixels height/width
hidden_size = 64
num_heads = 4
mlp_dim = 128


def build_ViT(input_shape):
    """ Builds the ViT model. """
    inputs = tf.keras.layers.Input(shape=input_shape)
    input_rescale = tf.keras.layers.Rescaling(scale=1/255)(inputs)  # rescale to the [0, 1] range

    # create the image patches encoder
    initial_patch_encodings = model_builder_functions.build_image_patches_encoder(patch_size, hidden_size, input_rescale)

    # build the ViT encoder
    encoder_out = model_builder_functions.vit_encoder(transformer_layers, mlp_dim, num_heads, initial_patch_encodings)

    # add the classification layer
    # im_representation = tf.reduce_mean(encoder_out, axis=1)  # (1,) or (1,2)
    backbone_extracted_features = tf.keras.layers.GlobalAveragePooling1D()(encoder_out)

    # note the linear activation
    logits = tf.keras.layers.Dense(units=len(CIFAR10_CLASSES), activation='linear', name='head',
                                   kernel_initializer=tf.keras.initializers.zeros)(backbone_extracted_features)

    vit_model = tf.keras.Model(inputs=inputs, outputs=logits)
    return vit_model