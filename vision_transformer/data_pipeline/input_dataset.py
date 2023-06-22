import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from sklearn.model_selection import train_test_split


def get_cifar10_raw_data():
    """ Load the raw CIFAR10 data. """
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    y_train = y_train.reshape((-1,))
    y_test = y_test.reshape((-1,))

    return x_train, y_train, x_test, y_test


def get_cifar10_data_splits(batch_size=128):
    """ Loads CIFAR10 data and creates train, val and test splits """
    x_train, y_train, x_test, y_test = get_cifar10_raw_data()
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42, shuffle=True)

    ds_train = create_tf_dataset(x_train, y_train, shuffle_buffer=40000, batch_size=batch_size)
    ds_val = create_tf_dataset(x_val, y_val, shuffle_buffer=0, batch_size=batch_size)
    ds_test = create_tf_dataset(x_test, y_test, shuffle_buffer=0, batch_size=batch_size)
    return ds_train, ds_val, ds_test


def create_tf_dataset(X, y, shuffle_buffer=0, batch_size=32):
    """Creates a tf dataset that is ready to be input into a model.

    Args:
        X (np.ndarray):
        y (np.ndarray):
        shuffle_buffer (int): if >0, then the dataset shuffling is applied
        batch_size (int):

    Returns:
        tf.data.Dataset
    """
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle_buffer:
        ds = ds.shuffle(buffer_size=shuffle_buffer)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds
