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


class EfficientDet(models.Model):

    def __init__(self, phi, num_classes=10, separable_conv=True, decode_outputs=True, **kwargs):
        super(EfficientDet, self).__init__(**kwargs)
        self.phi = phi
        self.num_classes = num_classes
        self.separable_conv = separable_conv
        self.decode_outputs = decode_outputs

        assert phi in range(7)
        input_size = input_image_resolution(phi)
        input_shape = (input_size, input_size, 1)
        image_input = layers.Input(input_shape)
        w_bifpn = bifpn_width(phi)
        d_bifpn = bifpn_depth(phi)
        w_head = heads_width(phi)
        d_head = heads_depth(phi)

        # create the efficientnet backbone
        self.feature_extractor = efficientnet(input_tensor=image_input, **efficientnet_params[phi])

        # weighted bifpn; efficient bidirectional cross-scale connections and weighted feature fusion
        self.bi_fpn = BiFeaturePyramid(n_blocks=d_bifpn, num_channels=w_bifpn)

        # bounding box model
        self.box_net = BoxNet(w_head, d_head, separable_conv=separable_conv, name='box_net')
        self.concat_regression = layers.Concatenate(axis=1, name='regression')

        # classification model
        self.class_net = ClassNet(w_head, d_head, num_classes=num_classes, separable_conv=separable_conv, name='class_net')
        self.concat_classification = layers.Concatenate(axis=1, name='classification')

        self.decoder = DecodePredictions(num_classes=num_classes, max_detections_per_class=10)

    def call(self, inputs, training=None, mask=None):
        features = self.feature_extractor(inputs)
        fpn_features = self.bi_fpn(features)
        regression = [self.box_net(feature) for feature in fpn_features]
        regression = self.concat_regression(regression)
        classification = [self.class_net(feature) for feature in fpn_features]
        classification = self.concat_classification(classification)

        if self.decode_outputs:
            return self.decoder([inputs, regression, classification])  # returns [boxes, scores, classes]
        else:
            return {'classification': classification, 'regression': regression}
