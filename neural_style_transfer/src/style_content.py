import tensorflow as tf


def gram_matrix(input_tensor):
    """Calculates the gram (style) matrix for an image (input_tensor).
    It basically computes dot products of all the values in a channel
    with all the values in the same or another channel. Thus, if you
    have an image of shape (nH x nW x nC), then the gram matrix will
    have shape (nC x nC). So it does not depend on the height or width
    of the input image!
    Finally it scales the gram matrix by the number of values in the
    input_tensor.

    Args:
        input_tensor (tf.Tensor):

    Returns:
        tf.Tensor
    """
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / (num_locations)


def clip_0_1(image):
    """Clips the image values to the range [0, 1] since this
    a scaled image should only have values in this range.

    Args:
        image (tf.Tensor):

    Returns:
        tf.Tensor
    """
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


def style_content_loss(outputs, content_targets, style_targets, alpha, beta):
    """Calculates the loss from the content targets, style targets
    and the generated image targets.

    Args:
        outputs (dict): the generated image targets
        content_targets (dict):
        style_targets (dict):
        alpha (float): weight for the content loss
        beta (float): weight for the style loss

    Returns:
        tf.Tensor
    """
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2)
                           for name in style_outputs.keys()])
    style_loss *= beta / len(style_outputs.keys())

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2)
                             for name in content_outputs.keys()])
    content_loss *= alpha / len(content_outputs.keys())
    loss = style_loss + content_loss
    return loss


def image_variation(img, weight=1):
    """This calculates the total variational loss. It is basically
    the sum of the absolute differences for neighboring pixel-values
    in the input images. It can be used as a regularization in the
    loss function of the style content loss.

    Args:
        img (tf.Tensor):
        weight (float):

    Returns:
        tf.Tensor
    """
    return tf.image.total_variation(img) * weight
