import tensorflow as tf
import numpy as np


def get_iou_vector(A, B):
    """Calculates the intersection over union (iou) and then checks for different
    thresholds whether the iou is high enough. Finally it computes the mean over
    all the thresholds.

    Args:
        A (np.ndarray):
        B (np.ndarray):

    Returns:
        float
    """
    batch_size = A.shape[0]
    thresholds = np.arange(0.5, 1, 0.05)
    metric = []
    for i in range(batch_size):
        t, p = A[i] > 0, B[i] > 0
        intersection = np.logical_and(t, p)
        union = np.logical_or(t, p)
        iou = (np.sum(intersection > 0) + 1e-10) / (np.sum(union > 0) + 1e-10)
        s = []
        for thresh in thresholds:
            s.append(iou > thresh)
        metric.append(np.mean(s))

    return np.mean(metric)


def iou_thr_05(label, pred):
    """calculates the intersection over union by applying a threshold
    of 0.5 to the predicted logits.

    Args:
        label (np.ndarray):
        pred (np.ndarray):

    Returns:
        float
    """
    return tf.compat.v1.py_func(get_iou_vector, [label, pred > 0.5], tf.float64)


def iou_thr_00(label, pred):
    """calculates the intersection over union by applying a threshold
    of zero to the predicted logits.

    Args:
        label (np.ndarray):
        pred (np.ndarray):

    Returns:
        float
    """
    return tf.compat.v1.py_func(get_iou_vector, [label, pred > 0], tf.float64)
