import tensorflow as tf


def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / (num_locations)


def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


def style_content_loss(outputs, content_targets, style_targets, alpha, beta):
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
