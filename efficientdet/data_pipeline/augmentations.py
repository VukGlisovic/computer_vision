import numpy as np
import tensorflow as tf
import imgaug.augmenters as aug_lib
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from efficientdet.constants import IMG_SHAPE


aug_affine = aug_lib.Affine(scale={"x": [0.9, 1.1], "y": [0.9, 1.1]},
                            translate_percent={"x": [-0.02, 0.02], "y": [-0.02, 0.02]},
                            rotate=[-5, 5])


def tf_affine_transform(image, boxes, labels):
    """Executes the affine_transform function by wrapping it inside a
     tf.numpy_function.
     All three inputs are required, since the image is augmented, the
     bounding box coordinates are shifted and some boxes with their
     labels can be removed because they're out of the image.

    Args:
        image (tf.Tensor): shape [H, W, C]
        boxes (tf.Tensor): variable number of boxes with shape [n_boxes, 4]
        labels (tf.Tensor): variable number of labels with shape [n_labels]
            where n_labels=n_boxes

    Returns:
        tuple[tf.Tensor, tf.Tensor, tf.Tensor]
    """
    image, boxes, labels = tf.numpy_function(affine_transform,
                                             [image, boxes, labels],
                                             Tout=(image.dtype, boxes.dtype, labels.dtype))
    image.set_shape(IMG_SHAPE)
    boxes.set_shape([None, 4])
    return image, boxes, labels


def affine_transform(image, boxes, labels):
    """This method executes the affine transformation. This method can be
    wrapped inside a tf.numpy_function such that it can be used in a tf
    dataset.

    Args:
        image (np.ndarray): shape [H, W, C]
        boxes (np.ndarray): variable number of boxes with shape [n_boxes, 4]
        labels (np.ndarray): variable number of labels with shape [n_labels]
            where n_labels=n_boxes

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    boxes = BoundingBoxesOnImage([BoundingBox(*coord) for coord in boxes], shape=image.shape)
    image, boxes = aug_affine(image=image, bounding_boxes=boxes)
    box_within_image_mask = np.array([~b.is_out_of_image(image, fully=True, partly=False) for b in boxes])
    # remove fully out of image bboxes, then clip the partially out of image bboxes, finally obtain numpy array from coordinates
    boxes = boxes.remove_out_of_image(fully=True, partly=False).clip_out_of_image().to_xyxy_array()
    labels = labels[box_within_image_mask]
    return image, boxes, labels
