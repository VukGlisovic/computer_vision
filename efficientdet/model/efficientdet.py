import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from functools import reduce

from efficientdet.model.custom_layers import ClipBoxes, RegressBoxes, FilterDetections, FastNormalizedFusion, ConvBlock, BiFPNFeatureFusion, BiFPNBlock
# from utils.anchors import anchors_for_shape
import numpy as np
from efficientdet.model.efficientnet_backbone import efficientnet
from efficientdet.model.efficientdet_heads import BoxNet, ClassNet
from efficientdet.model.anchors2 import anchors_for_shape


efficientnet_params = [
    dict(width_coefficient=1.0, depth_coefficient=1.0, dropout_rate=0.2, model_name='efficientnet-b0', return_fpn_features=True),
    dict(width_coefficient=1.0, depth_coefficient=1.1, dropout_rate=0.2, model_name='efficientnet-b1', return_fpn_features=True),
    dict(width_coefficient=1.1, depth_coefficient=1.2, dropout_rate=0.2, model_name='efficientnet-b2', return_fpn_features=True),
    dict(width_coefficient=1.2, depth_coefficient=1.4, dropout_rate=0.2, model_name='efficientnet-b3', return_fpn_features=True),
    dict(width_coefficient=1.4, depth_coefficient=1.8, dropout_rate=0.2, model_name='efficientnet-b4', return_fpn_features=True),
    dict(width_coefficient=1.6, depth_coefficient=2.2, dropout_rate=0.2, model_name='efficientnet-b5', return_fpn_features=True),
    dict(width_coefficient=1.8, depth_coefficient=2.6, dropout_rate=0.2, model_name='efficientnet-b6', return_fpn_features=True),
    dict(width_coefficient=2.0, depth_coefficient=3.1, dropout_rate=0.2, model_name='efficientnet-b7', return_fpn_features=True),
]


def bifpn_width(phi):
    return int(64 + 1.35**phi)


def bifpn_depth(phi):
    return 3 + phi


def heads_width(phi):
    return bifpn_width(phi)


def heads_depth(phi):
    return 3 + int(phi / 3)


def input_image_resolution(phi):
    return 512 + phi * 128


def build_wBiFPN(features, num_channels, id):
    if id == 0:
        _, _, C3, C4, C5 = features
        P3_in = C3
        P4_in = C4
        P5_in = C5
        P6_in = layers.Conv2D(num_channels, kernel_size=1, padding='same', name='resample_p6/conv2d')(C5)
        P6_in = layers.BatchNormalization(name='resample_p6/bn')(P6_in)
        P6_in = layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name='resample_p6/maxpool')(P6_in)
        P7_in = layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name='resample_p7/maxpool')(P6_in)
        P3_out, P4_out, P5_out, P6_out, P7_out = BiFPNBlock(num_channels, add_conv_blocks=True, id=id)([P3_in, P4_in, P5_in, P6_in, P7_in])
    else:
        # P3_in, P4_in, P5_in, P6_in, P7_in = features
        P3_out, P4_out, P5_out, P6_out, P7_out = BiFPNBlock(num_channels, add_conv_blocks=False, id=id)(features)
    return P3_out, P4_out, P5_out, P6_out, P7_out


def efficientdet(phi, num_classes=10, num_anchors=9,
                 score_threshold=0.01, anchor_parameters=None, separable_conv=True):
    assert phi in range(7)
    input_size = input_image_resolution(phi)
    input_shape = (input_size, input_size, 1)
    image_input = layers.Input(input_shape)
    w_bifpn = bifpn_width(phi)
    d_bifpn = bifpn_depth(phi)
    w_head = heads_width(phi)
    d_head = heads_depth(phi)
    features = efficientnet(input_tensor=image_input, **efficientnet_params[phi])

    # weighted bifpn
    fpn_features = features
    for i in range(d_bifpn):
        fpn_features = build_wBiFPN(fpn_features, w_bifpn, i)

    # bounding box model
    box_net = BoxNet(w_head, d_head, num_anchors=num_anchors, separable_conv=separable_conv, name='box_net')
    regression = [box_net(feature) for feature in fpn_features]
    regression = layers.Concatenate(axis=1, name='regression')(regression)
    # classification model
    class_net = ClassNet(w_head, d_head, num_classes=num_classes, num_anchors=num_anchors,
                         separable_conv=separable_conv, name='class_net')
    classification = [class_net(feature) for feature in fpn_features]
    classification = layers.Concatenate(axis=1, name='classification')(classification)

    model = models.Model(inputs=[image_input], outputs=[classification, regression], name='efficientdet')

    # apply predicted regression to anchors
    anchors = anchors_for_shape((input_size, input_size), anchor_params=anchor_parameters)
    anchors_input = np.expand_dims(anchors, axis=0)
    boxes = RegressBoxes(name='boxes')([anchors_input, regression[..., :4]])
    boxes = ClipBoxes(name='clipped_boxes')([image_input, boxes])

    # filter detections (apply NMS / score threshold / select top-k)
    detections = FilterDetections(
        name='filtered_detections',
        score_threshold=score_threshold
    )([boxes, classification])

    prediction_model = models.Model(inputs=[image_input], outputs=detections, name='efficientdet_p')
    return model, prediction_model
