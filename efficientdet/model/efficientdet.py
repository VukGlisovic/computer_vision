from tensorflow.keras import layers
from tensorflow.keras import models

from efficientdet.model.custom_layers import BiFeaturePyramid
from efficientdet.model.efficientnet_backbone import efficientnet
from efficientdet.model.efficientdet_heads import BoxNet, ClassNet
from efficientdet.data_pipeline.decode_predictions import DecodePredictions


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


def efficientdet(phi, num_classes=10, num_anchors=9, separable_conv=True):
    # determine network size parameters
    assert phi in range(7)
    input_size = input_image_resolution(phi)
    input_shape = (input_size, input_size, 1)
    image_input = layers.Input(input_shape)
    w_bifpn = bifpn_width(phi)
    d_bifpn = bifpn_depth(phi)
    w_head = heads_width(phi)
    d_head = heads_depth(phi)

    # create the efficientnet backbone
    features = efficientnet(input_tensor=image_input, **efficientnet_params[phi])

    # weighted bifpn; efficient bidirectional cross-scale connections and weighted feature fusion
    fpn_features = BiFeaturePyramid(n_blocks=d_bifpn, num_channels=w_bifpn)(features)

    # bounding box model
    box_net = BoxNet(w_head, d_head, num_anchors=num_anchors, separable_conv=separable_conv, name='box_net')
    regression = [box_net(feature) for feature in fpn_features]
    regression = layers.Concatenate(axis=1, name='regression')(regression)

    # classification model
    class_net = ClassNet(w_head, d_head, num_classes=num_classes, num_anchors=num_anchors, separable_conv=separable_conv, name='class_net')
    classification = [class_net(feature) for feature in fpn_features]
    classification = layers.Concatenate(axis=1, name='classification')(classification)

    model = models.Model(inputs=[image_input], outputs=[classification, regression], name='efficientdet')

    detections = DecodePredictions(num_classes=num_classes, max_detections_per_class=10)([image_input, regression, classification])
    prediction_model = models.Model(inputs=[image_input], outputs=detections, name='efficientdet_p')
    return model, prediction_model
