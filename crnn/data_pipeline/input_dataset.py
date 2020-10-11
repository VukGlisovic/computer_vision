import tensorflow as tf
from crnn.data_pipeline.image_dataset import create_images_dataset
from crnn.data_pipeline.label_dataset import create_label_dataset


def input_fn(df, epochs=None, batch_size=32, shuffle_buffer=64, augment=False):
    ds_images = create_images_dataset(df, augment)
    ds_labels = create_label_dataset(df)
    dataset = tf.data.Dataset.zip((ds_images, ds_labels))
    if shuffle_buffer:
        dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(epochs)
    return dataset