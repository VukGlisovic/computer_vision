import string
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

from efficientdet.data_pipeline.utils import xyxy_to_xywh, xywh_to_xyxy
from efficientdet.data_pipeline.anchors import AnchorBox
from efficientdet.data_pipeline.target_encoder import TargetEncoder
from efficientdet.model.efficientdet import efficientdet
from efficientdet.model.losses import smooth_l1, focal


def main():
    model, model_prediction = efficientdet(phi=0)
    model.summary()

    def read_csv(path):
        df = pd.read_csv(path, dtype={'img_path': str, 'x1': 'int32', 'y1': 'int32', 'x2': 'int32', 'y2': 'int32',
                                      'label': 'int32'})
        print(df.shape)
        return df

    df_train = read_csv('../data/train.csv')
    df_test = read_csv('../data/test.csv')

    def load_image(path):
        image_string = tf.io.read_file(path)
        image = tf.io.decode_jpeg(image_string, channels=1)
        return image

    def preprocess_image(img):
        img = img / 255
        return img

    def load_and_preprocess_image(path):
        img = load_image(path)
        img_preprocessed = preprocess_image(img)
        img_preprocessed.set_shape((512, 512, 1))
        return img_preprocessed

    def prepare_bbox(bbox):
        return tf.expand_dims(bbox, axis=0)

    def prepare_label(label):
        return tf.expand_dims(label, axis=0)

    def name_targets(img, box_targets, cls_targets):
        return img, {'regression': box_targets, 'classification': cls_targets}

    ds_image_paths = tf.data.Dataset.from_tensor_slices(df_train['img_path'].values)
    ds_image_paths = ds_image_paths.map(load_and_preprocess_image)

    ds_bbox = tf.data.Dataset.from_tensor_slices(df_train[['x1', 'y1', 'x2', 'y2']].values)
    ds_bbox = ds_bbox.map(xyxy_to_xywh)
    ds_bbox = ds_bbox.map(prepare_bbox)

    ds_labels = tf.data.Dataset.from_tensor_slices(df_train['label'].values)
    ds_labels = ds_labels.map(prepare_label)

    # combine datasets in one dataset
    ds = tf.data.Dataset.zip((ds_image_paths, ds_bbox, ds_labels))
    # target encoder needs both box and label information to create target encodings
    target_encoder = TargetEncoder()
    ds = ds.map(target_encoder.encode_sample)
    ds = ds.batch(8)
    ds = ds.map(name_targets)

    adam = tf.keras.optimizers.Adam()
    losses = {'regression': smooth_l1(), 'classification': focal()}

    def scheduler(epoch, lr):
        return lr * tf.math.exp(-0.1)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint('../data/results/checkpoints/efficientdet-{epoch:02d}.hdf5', save_best_only=False, save_weights_only=True),
        tf.keras.callbacks.LearningRateScheduler(scheduler)
    ]

    model.compile(optimizer=adam, loss=losses)

    history = model.fit(ds, epochs=1, callbacks=callbacks)


if __name__ == '__main__':
    main()
