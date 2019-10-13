import tensorflow as tf
import numpy as np


def get_iou_vector(A, B):
    batch_size = A.shape[0]
    metric = []
    for i in range(batch_size):
        t, p = A[i] > 0, B[i] > 0
        intersection = np.logical_and(t, p)
        union = np.logical_or(t, p)
        iou = (np.sum(intersection > 0) + 1e-10) / (np.sum(union > 0) + 1e-10)
        thresholds = np.arange(0.5, 1, 0.05)
        s = []
        for thresh in thresholds:
            s.append(iou > thresh)
        metric.append(np.mean(s))

    return np.mean(metric)


def iou(label, pred):
    return tf.compat.v1.py_func(get_iou_vector, [label, pred > 0.5], tf.dtypes.float64)
