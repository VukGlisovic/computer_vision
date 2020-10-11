import tensorflow as tf
from functools import partial


def apply_augmentations(tf_dataset):
    tf_dataset = tf_dataset.map(tf_random_rotate_image(rg=3))
    tf_dataset = tf_dataset.map(tf_random_shear_image(intensity=3))
    tf_dataset = tf_dataset.map(tf_random_zoom_image(zoom_range_lower=0.96, zoom_range_upper=1.04))
    tf_dataset = tf_dataset.map(tf_random_shift_image(wrg=0.01, hrg=0.01))
    tf_dataset.map(tf_random_saturation(lower=0.9, upper=1.1))
    tf_dataset.map(tf_random_brightness(max_delta=0.02))
    tf_dataset.map(tf_random_contrast(lower=0.9, upper=1.1))
    # correct if any numbers fall out of range
    tf_dataset = tf_dataset.map(clip_image_0_1)
    return tf_dataset


def clip_image_0_1(img):
    return tf.clip_by_value(img, 0, 1)


def tf_random_saturation(lower, upper):
    return partial(tf.image.random_saturation, lower=lower, upper=upper)


def tf_random_brightness(max_delta):
    return partial(tf.image.random_brightness, max_delta=max_delta)


def tf_random_contrast(lower, upper):
    return partial(tf.image.random_contrast, lower=lower, upper=upper)


def tf_random_rotate_image(rg):

    def wrapper_random_rotate_image(image):

        def random_rotate_image(img):
            return tf.keras.preprocessing.image.random_rotation(
                img.numpy(),
                rg=rg,
                row_axis=0,
                col_axis=1,
                channel_axis=2
            )

        [image,] = tf.py_function(random_rotate_image, [image], [tf.float32])
        return image

    return wrapper_random_rotate_image


def tf_random_shear_image(intensity):

    def wrapper_random_shear_image(image):

        def random_shear_image(img):
            return tf.keras.preprocessing.image.random_shear(
                img.numpy(),
                intensity=intensity,
                row_axis=0,
                col_axis=1,
                channel_axis=2
            )

        [image, ] = tf.py_function(random_shear_image, [image], [tf.float32])
        return image

    return wrapper_random_shear_image


def tf_random_zoom_image(zoom_range_lower, zoom_range_upper):

    def wrapper_random_zoom_image(image):

        def random_zoom_image(img):
            return tf.keras.preprocessing.image.random_zoom(
                img.numpy(),
                zoom_range=(zoom_range_lower, zoom_range_upper),
                row_axis=0,
                col_axis=1,
                channel_axis=2
            )

        [image, ] = tf.py_function(random_zoom_image, [image], [tf.float32])
        return image

    return wrapper_random_zoom_image


def tf_random_shift_image(wrg, hrg):

    def wrapper_random_shift_image(image):

        def random_shift_image(img):
            return tf.keras.preprocessing.image.random_shift(
                img.numpy(),
                wrg=wrg,
                hrg=hrg,
                row_axis=0,
                col_axis=1,
                channel_axis=2
            )

        [image, ] = tf.py_function(random_shift_image, [image], [tf.float32])
        return image

    return wrapper_random_shift_image
