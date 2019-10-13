from unet.model.architecture import *
from unet.model.constants import *


def get_model():
    input_image = keras.layers.Input(IMAGE_SHAPE, name='image')
    unet_model = create_unet_model(input_image, batchnorm=False)
    return unet_model


if __name__ == '__main__':
    test = get_model()
