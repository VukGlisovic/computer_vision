import tensorflow as tf

from blazeface.dataset import utils


def randomly_apply_augmentations(img, bboxes, lmarks):
    """Randomly choose which and apply augmentation methods.

    Args:
        img (tf.Tensor): shape [height, width, 3]
        bboxes (tf.Tensor): shape [n_bboxes, [x1, y1, x2, y2]] in the [0, 1] range.
        lmarks (tf.Tensor): shape [n_bboxes, n_landmarks, [x, y]] in the [0, 1] range.

    Returns:
        tuple[tf.Tensor, tf.Tensor, tf.Tensor]: the image, bounding boxes and landmarks
    """
    # color augmentation
    color_methods = [random_brightness, random_contrast, random_hue, random_saturation]
    # geometric operations
    geometric_methods = [random_image_crop]
    # randomly apply augmentations
    for augmentation_method in geometric_methods + color_methods:
        img, bboxes, lmarks = randomly_apply_method(augmentation_method, img, bboxes, lmarks)
    img = tf.clip_by_value(img, 0., 1.)
    return img, bboxes, lmarks


def get_random_bool(p=0.9):
    """Gets a random boolean. p indicates the probability of obtaining
    a True bool value as output.

    Args:
        p (float):

    Returns:
        tf.Tensor: boolean value
    """
    return tf.less(tf.random.uniform((), dtype=tf.float32), p)


def randomly_apply_method(fnc, img, bboxes, lmarks, *args):
    """Randomly applying given method to image and ground truth boxes.

    Args:
        fnc (Callable):
        img (tf.Tensor): shape [height, width, 3]
        bboxes (tf.Tensor): shape [n_bboxes, [x1, y1, x2, y2]]
        lmarks (tf.Tensor): shape [n_bboxes, n_landmarks, [x, y]]
        *args: any arguments that should be passed to fnc

    Returns:
        tuple[tf.Tensor, tf.Tensor, tf.Tensor]: the image, bounding boxes and landmarks
    """
    return tf.cond(
        get_random_bool(),
        lambda: fnc(img, bboxes, lmarks, *args),
        lambda: (img, bboxes, lmarks)
    )

###########################
### Color augmentations ###
###########################

def random_brightness(img, gt_boxes, gt_landmarks, max_delta=0.12):
    """Apply random brightness augmentation.

    Args:
        img (tf.Tensor):
        gt_boxes (tf.Tensor):
        gt_landmarks (tf.Tensor):
        max_delta (float):

    Returns:
        tuple[tf.Tensor, tf.Tensor, tf.Tensor]: the image, bounding boxes and landmarks
    """
    return tf.image.random_brightness(img, max_delta), gt_boxes, gt_landmarks


def random_contrast(img, gt_boxes, gt_landmarks, lower=0.5, upper=1.5):
    """Apply random contrast augmentation.

    Args:
        img (tf.Tensor):
        gt_boxes (tf.Tensor):
        gt_landmarks (tf.Tensor):
        lower (float):
        upper (float):

    Returns:
        tuple[tf.Tensor, tf.Tensor, tf.Tensor]: the image, bounding boxes and landmarks
    """
    return tf.image.random_contrast(img, lower, upper), gt_boxes, gt_landmarks


def random_hue(img, gt_boxes, gt_landmarks, max_delta=0.08):
    """Apply random hue augmentation.

    Args:
        img (tf.Tensor):
        gt_boxes (tf.Tensor):
        gt_landmarks (tf.Tensor):
        max_delta (float):

    Returns:
        tuple[tf.Tensor, tf.Tensor, tf.Tensor]: the image, bounding boxes and landmarks
    """
    return tf.image.random_hue(img, max_delta), gt_boxes, gt_landmarks


def random_saturation(img, gt_boxes, gt_landmarks, lower=0.5, upper=1.5):
    """Apply random saturation augmentation.

    Args:
        img (tf.Tensor):
        gt_boxes (tf.Tensor):
        gt_landmarks (tf.Tensor):
        lower (float):
        upper (float):

    Returns:
        tuple[tf.Tensor, tf.Tensor, tf.Tensor]: the image, bounding boxes and landmarks
    """
    return tf.image.random_saturation(img, lower, upper), gt_boxes, gt_landmarks

###############################
### Geometric augmentations ###
###############################

def get_random_min_overlap():
    """Get random min overlap value for tf.image.sample_distorted_bounding_box.

    Returns:
        tf.Tensor
    """
    overlaps = tf.constant([0.1, 0.3, 0.5, 0.7, 0.9], dtype=tf.float32)
    i = tf.random.uniform((), minval=0, maxval=tf.shape(overlaps)[0], dtype=tf.int32)
    return overlaps[i]


def pad_image(img, gt_boxes, gt_landmarks, height, width):
    """Increases the size of the input img by padding with the image
    moments. Afterwards adjusts the bounding box and landmark coordinates.

    Args:
        img (tf.Tensor): shape [height, width, 3]
        gt_boxes (tf.Tensor): shape [n_bboxes, [x1, y1, x2, y2]]
        gt_landmarks (tf.Tensor): [n_bboxes, n_landmarks, [x, y]]
        height (tf.Tensor): height of the image
        width (tf.Tensor): width of the image

    Returns:
        tuple[tf.Tensor, tf.Tensor, tf.Tensor]: modified image with adjusted bounding boxes and landmarks
    """
    expansion_ratio = tf.random.uniform((), minval=1, maxval=4, dtype=tf.float32)
    final_height, final_width = tf.round(height * expansion_ratio), tf.round(width * expansion_ratio)
    # get random padding sizes for the height and width
    pad_left = tf.round(tf.random.uniform((), minval=0, maxval=final_width - width, dtype=tf.float32))
    pad_top = tf.round(tf.random.uniform((), minval=0, maxval=final_height - height, dtype=tf.float32))
    pad_right = final_width - (width + pad_left)
    pad_bottom = final_height - (height + pad_top)
    # pad img with the img moments
    mean, _ = tf.nn.moments(img, [0, 1])
    expanded_image = tf.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), constant_values=-1)
    expanded_image = tf.where(expanded_image == -1, mean, expanded_image)
    # adjust the bounding box and landmark coordinates
    min_max = tf.stack([-pad_top, -pad_left, pad_bottom + height, pad_right + width], axis=-1) / [height, width, height, width]
    modified_gt_boxes = utils.adjust_bbox_coordinates(gt_boxes, min_max)
    modified_gt_landmarks = utils.adjust_landmark_coordinates(gt_landmarks, min_max)
    return expanded_image, modified_gt_boxes, modified_gt_landmarks


def random_image_crop(img, bboxes, lmarks):
    """Expands the image and then takes a random crop from the input img
    and adjusts bounding box and landmark coordinates to the cropped image.

    After this operation some bounding boxes and landmarks may be removed
    from the img. These bounding boxes and landmarks are not excluded from
    the output, only the coordinates are changed to zero.

    Args:
        img (tf.Tensor): shape [height, width, 3]
        bboxes (tf.Tensor): shape [n_bboxes, [x1, y1, x2, y2]] in the [0, 1] range.
        lmarks (tf.Tensor): shape [n_bboxes, n_landmarks, [x, y]] in the [0, 1] range.

    Returns:
        tuple[tf.Tensor, tf.Tensor, tf.Tensor]: the image, bounding boxes and landmarks
    """
    img_shape = tf.cast(tf.shape(img), dtype=tf.float32)
    orig_height, orig_width = img_shape[0], img_shape[1]
    # apply random padding to the img
    img, bboxes, lmarks = randomly_apply_method(pad_image, img, bboxes, lmarks, orig_height, orig_width)
    # randomly calculate which part of the expanded img to crop
    min_overlap = get_random_min_overlap()
    begin, size, new_boundaries = tf.image.sample_distorted_bounding_box(
        tf.shape(img),
        bounding_boxes=tf.expand_dims(bboxes, 0),
        min_object_covered=min_overlap)
    # crop the img and adjust the bounding box and landmark coordinates
    img = tf.slice(img, begin, size)
    img = tf.image.resize(img, (orig_height, orig_width))
    bboxes = utils.adjust_bbox_coordinates(bboxes, new_boundaries[0, 0])
    lmarks = utils.adjust_landmark_coordinates(lmarks, new_boundaries[0, 0])

    return img, bboxes, lmarks
