from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization, Activation, Reshape, \
    Bidirectional, LSTM, Dense


def build_model(num_classes, channels=1):
    """The original architecture from the CRNN paper.
    """
    input_image = keras.Input(shape=(256, 32, channels))
    x = Conv2D(64, 3, padding='same', activation='relu')(input_image)
    x = MaxPool2D(pool_size=2, padding='same')(x)

    x = Conv2D(128, 3, padding='same', activation='relu')(x)
    x = MaxPool2D(pool_size=2, padding='same')(x)

    x = Conv2D(256, 3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, 3, padding='same', activation='relu')(x)
    x = MaxPool2D(pool_size=2, strides=(1, 2), padding='same')(x)

    x = Conv2D(512, 3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(512, 3, padding='same', activation='relu')(x)
    x = MaxPool2D(pool_size=2, strides=(1, 2), padding='same')(x)

    x = Conv2D(512, 2, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    output_shape = x.get_shape()
    target_shape = (int(output_shape[1]), int(output_shape[2] * output_shape[3]))
    x = Reshape(target_shape)(x)

    x = Dense(64, activation='relu')(x)

    x = Bidirectional(LSTM(units=256, return_sequences=True), merge_mode='sum')(x)
    x = Bidirectional(LSTM(units=256, return_sequences=True), merge_mode='sum')(x)
    x = Dense(units=num_classes)(x)
    return keras.Model(inputs=input_image, outputs=x, name='CRNN')
