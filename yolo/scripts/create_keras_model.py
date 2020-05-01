import os
import importlib
from keras.utils import plot_model


# import vendor module
keras_yolo = importlib.import_module('yolo.vendor.keras-yolo3.yolo3_one_file_to_detect_them_all')


def create_keras_model():
    # define the model
    model = keras_yolo.make_yolov3_model()
    # load the model weights
    weight_reader = keras_yolo.WeightReader('../models/MSCOCO/yolov3.weights')
    # set the model weights into the model
    weight_reader.load_weights(model)
    # save the model to file
    model.save('../models/MSCOCO/model_yolov3.h5')
    return model


def plot_model_graph(model):
    data_dir = '../data'
    os.makedirs(data_dir, exist_ok=True)
    plot_model(model,
               to_file=os.path.join(data_dir, 'model_yolov3.png'),
               show_shapes=True,
               show_layer_names=True)


def main():
    model = create_keras_model()
    plot_model_graph(model)


if __name__ == '__main__':
    main()
