import pandas as pd
import tensorflow as tf
from efficientdet.constants import IMG_SHAPE
from efficientdet.data_pipeline.utils import xyxy_to_xywh
from efficientdet.data_pipeline.target_encoder import TargetEncoder


def read_csv(path):
    """Helper function to easily read the data into a pandas dataframe.

    Args:
        path (str):

    Returns:
        pd.DataFrame
    """
    dtypes = {'img_path': str, 'x1': 'int32', 'y1': 'int32', 'x2': 'int32', 'y2': 'int32', 'label': 'int32'}
    df = pd.read_csv(path, dtype=dtypes)
    return df


def load_image(path):
    """Loads an image from disk.

    Args:
        path (str):

    Returns:
        tf.Tensor
    """
    image_string = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image_string, channels=1)
    return image


def preprocess_image(img):
    """Normalizes an image.

    Args:
        img (tf.Tensor):

    Returns:
        tf.Tensor
    """
    img = img / 255
    return img


def load_and_preprocess_image(path):
    """Loads and preprocesses one image.

    Args:
        path (str):

    Returns:
        tf.Tensor
    """
    img = load_image(path)
    img_preprocessed = preprocess_image(img)
    img_preprocessed.set_shape(IMG_SHAPE)
    return img_preprocessed


def expand_dims(bbox):
    """Adds a dimension along the first axis.

    Args:
        bbox (tf.Tensor):

    Returns:
        tf.Tensor
    """
    return tf.expand_dims(bbox, axis=0)


def name_targets(img, box_targets, cls_targets):
    """Gives names to the regression and classification targets such
    that the model can distinguish between the two. The img is simply
    passed on.

    Args:
        img (tf.Tensor):
        box_targets (tf.Tensor):
        cls_targets (tf.Tensor):

    Returns:
        tf.Tensor
    """
    return img, {'regression': box_targets, 'classification': cls_targets}


def create_images_dataset(df):
    """Creates a tf dataset for the input images.

    Args:
        df (pd.DataFrame):

    Returns:
        tf.data.Dataset
    """
    ds_image = tf.data.Dataset.from_tensor_slices(df['img_path'].values)
    ds_image = ds_image.map(load_and_preprocess_image)
    return ds_image


def create_bbox_dataset(df):
    """Creates a tf dataset for the bounding box targets.

    Args:
        df (pd.DataFrame):

    Returns:
        tf.data.Dataset
    """
    ragged_tensor_coordinates = tf.ragged.constant(df['coordinates'].values)
    ds_bbox = tf.data.Dataset.from_tensor_slices(ragged_tensor_coordinates)
    ds_bbox = ds_bbox.map(xyxy_to_xywh)
    ds_bbox = ds_bbox.map(lambda x: x.to_tensor())
    return ds_bbox


def create_labels_dataset(df):
    """Creates a tf dataset for the class labels.

    Args:
        df (pd.DataFrame):

    Returns:
        tf.data.Dataset
    """
    ragged_tensor_labels = tf.ragged.constant(df['label'].values)
    ds_labels = tf.data.Dataset.from_tensor_slices(ragged_tensor_labels)
    return ds_labels


def create_combined_dataset(path, batch_size=8):
    """Creates one tf dataset that combines the images tf dataset,
    the regression tf dataset and the classification tf dataset.

    Args:
        path (str):
        batch_size (int):

    Returns:
        tf.data.Dataset
    """
    # read and group all boxes on one image
    df = read_csv(path)
    df['coordinates'] = df[['x1', 'y1', 'x2', 'y2']].values.tolist()
    df_grouped = df.groupby('img_path')[['coordinates', 'label']].agg(list)
    df_grouped.reset_index(drop=False, inplace=True)
    # create separate datasets
    ds_image_paths = create_images_dataset(df_grouped)
    ds_bbox = create_bbox_dataset(df_grouped)
    ds_labels = create_labels_dataset(df_grouped)
    # combine datasets in one dataset
    ds = tf.data.Dataset.zip((ds_image_paths, ds_bbox, ds_labels))
    # target encoder needs both box and label information to create target encodings
    target_encoder = TargetEncoder()
    ds = ds.map(target_encoder.encode_sample)
    ds = ds.batch(batch_size)
    ds = ds.map(name_targets)
    return ds
