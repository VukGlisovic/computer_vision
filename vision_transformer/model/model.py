import tensorflow as tf

from vision_transformer.constants import CIFAR10_CLASSES
from vision_transformer.model import custom_layers


######################################
# hyperparameter section
######################################
transformer_layers = 6
patch_size = 4
hidden_size = 64
num_heads = 4
mlp_dim = 128

######################################

rescale_layer = tf.keras.Sequential([tf.keras.layers.experimental.preprocessing.Rescaling(1./255)])


def build_ViT(input_shape):
    inputs = tf.keras.layers.Input(shape=input_shape)
    # rescaling (normalizing pixel val between 0 and 1)
    rescale = rescale_layer(inputs)
    # generate patches with conv layer
    patch_encodings = custom_layers.build_image_patches_encoder(patch_size, hidden_size, rescale)

    ######################################
    # ready for the transformer blocks
    ######################################
    encoder_out = custom_layers.vit_encoder(transformer_layers, mlp_dim, num_heads, patch_encodings)

    #####################################
    #  final part (mlp to classification)
    #####################################
    #encoder_out_rank = int(tf.experimental.numpy.ndim(encoder_out))
    im_representation = tf.reduce_mean(encoder_out, axis=1)  # (1,) or (1,2)
    # similar to the GAP, this is from original Google GitHub

    logits = tf.keras.layers.Dense(units=len(CIFAR10_CLASSES), name='head', kernel_initializer=tf.keras.initializers.zeros)(im_representation)
    # !!! important !!! activation is linear

    final_model = tf.keras.Model(inputs = inputs, outputs = logits)
    return final_model