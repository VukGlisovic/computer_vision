import logging
import tensorflow as tf
import tensorflow.keras.backend as K


def lovasz_grad(gt_sorted):
    """Computes gradient of the Lovasz extension w.r.t sorted errors
    See Algorithm 1 in https://arxiv.org/pdf/1705.08790.pdf

    Args:
        gt_sorted (tf.Tensor): ground truths sorted

    Returns:
        tf.Tensor
    """
    gts = tf.reduce_sum(gt_sorted)
    intersection = gts - tf.cumsum(gt_sorted)
    union = gts + tf.cumsum(1. - gt_sorted)
    jaccard = 1. - intersection / union
    jaccard = tf.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)
    return jaccard


def lovasz_hinge_flat(logits, labels):
    """Binary Lovasz hinge loss.

    Args:
        logits (tf.Tensor): logits at each prediction (between -\infty and +\infty)
        labels (tf.Tensor): binary ground truth labels (0 or 1)
        ignore (Union[str, None]): label to ignore

    Returns:
        tf.Tensor
    """
    def compute_loss():
        labelsf = tf.cast(labels, logits.dtype)
        signs = 2. * labelsf - 1.
        errors = 1. - logits * tf.stop_gradient(signs)
        errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0], name="descending_sort")
        gt_sorted = tf.gather(labelsf, perm)
        grad = lovasz_grad(gt_sorted)
        loss = tf.tensordot(tf.nn.elu(errors_sorted), tf.stop_gradient(grad), 1, name="loss_non_void")
        return loss

    # deal with the void prediction case (only void pixels)
    loss = tf.cond(tf.equal(tf.shape(logits)[0], 0),
                   lambda: tf.reduce_sum(logits) * 0.,
                   compute_loss,
                   name="loss")
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """Flattens predictions in the batch (binary case). Remove
    labels equal to 'ignore'

    Args:
        scores (tf.Tensor):
        labels (tf.Tensor):
        ignore (Union[str, None]):

    Returns:
        tuple[tf.Tensor, tf.Tensor]
    """
    scores = tf.reshape(scores, (-1,))
    labels = tf.reshape(labels, (-1,))
    if ignore is None:
        return scores, labels
    valid = tf.not_equal(labels, ignore)
    vscores = tf.boolean_mask(scores, valid, name='valid_scores')
    vlabels = tf.boolean_mask(labels, valid, name='valid_labels')
    return vscores, vlabels


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """Binary Lovasz hinge loss.

    Args:
        logits (tf.Tensor): [batch, height, width] tensor, logits at
            each pixel (between -\infty and +\infty)
        labels (tf.Tensor): [batch, height, width] tensor, binary ground
            truth masks (0 or 1)
        per_image (bool): compute the loss per image instead of per batch
        ignore (Union[str, None]): void class id

    Returns:
        tf.Tensor
    """
    if per_image:
        def treat_image(log_lab):
            log, lab = log_lab
            log, lab = tf.expand_dims(log, 0), tf.expand_dims(lab, 0)
            log, lab = flatten_binary_scores(log, lab, ignore)
            return lovasz_hinge_flat(log, lab)

        losses = tf.map_fn(treat_image, (logits, labels), dtype=tf.float32)
        loss = tf.reduce_mean(losses)
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_loss(y_true, y_pred):
    """Lovasz Loss function according to https://arxiv.org/pdf/1705.08790.pdf

    Args:
        y_true (tf.Tensor):
        y_pred (tf.Tensor):

    Returns:
        tf.Tensor
    """
    y_true, y_pred = K.cast(K.squeeze(y_true, -1), 'int32'), K.cast(K.squeeze(y_pred, -1), 'float32')
    # logits = K.log(y_pred / (1. - y_pred))
    logits = y_pred  # Jiaxin
    loss = lovasz_hinge(logits, y_true, per_image=True, ignore=None)
    return loss


def get_loss_function(loss):
    """Tries to retrieve the requested loss function. Raises an error
    if it cannot find it.

    Args:
        loss (str):

    Returns:
        Callable
    """
    try:
        loss_fnc = globals()[loss]
        return loss_fnc
    except KeyError:
        logging.info("No loss named '%s' in custom losses module.")
    try:
        loss_fnc = getattr(tf.keras.losses, loss)
        return loss_fnc
    except AttributeError:
        logging.info("No loss named '%s' in keras.losses module.")
    raise ImportError("Could not find loss function '{}'.".format(loss))
