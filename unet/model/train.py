from unet.model.constants import *
from unet.model.preprocessing import input_fn, load_data
from unet.model.architecture import *
from unet.model.metrics import iou
from tensorflow import keras
from sklearn.model_selection import train_test_split


def get_model():
    input_image = keras.layers.Input(IMAGE_SHAPE, name='image')
    unet_model = create_unet_model(input_image, batchnorm=False)

    model = keras.Model(input_image, unet_model)
    optimizer = keras.optimizers.adam(lr=0.01)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=[iou])
    return model


def train_and_validate(model):
    Xdata, ydata = load_data()
    Xtrain, Xvalid, ytrain, yvalid = train_test_split(Xdata, ydata)
    train_dataset = input_fn(Xtrain, ytrain, epochs=2, batch_size=32, shuffle_buffer=300)
    valid_dataset = input_fn(Xvalid, yvalid, epochs=None, batch_size=128, shuffle_buffer=None)

    model_checkpoint = keras.callbacks.ModelCheckpoint('unet_saved_model', monitor='iou', mode='max', save_best_only=True, verbose=1)
    model.fit(train_dataset,
              validation_data=valid_dataset,
              callbacks=[model_checkpoint],
              verbose=2)


if __name__ == '__main__':
    model = get_model()
    train_and_validate(model)
