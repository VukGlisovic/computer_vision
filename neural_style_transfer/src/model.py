import tensorflow as tf
from neural_style_transfer.src import style_content


class StyleContentModelVGG(tf.keras.models.Model):
    """Creates a keras model for neural style transfer by using
    a pretrained VGG network.

    Args:
        style_layers (list[str]): the layer names of the vgg network
            from which to use the output for the style loss.
        content_layers (list[str]): the layer names of the vgg network
            from which to use the output for the content loss.
    """

    def __init__(self, style_layers, content_layers):
        super(StyleContentModelVGG, self).__init__()
        self.vgg = self.extract_vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)

    def call(self, inputs):
        """When called, it computes the gram matrix for the style
        part of the image.
        It expects float input in [0, 1].

        Args:
            inputs (tf.Tensor): an image

        Returns:
            dict
        """
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])

        style_outputs = [style_content.gram_matrix(style_output)
                         for style_output in style_outputs]

        content_dict = {content_name: value
                        for content_name, value
                        in zip(self.content_layers, content_outputs)}

        style_dict = {style_name: value
                      for style_name, value
                      in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}

    @staticmethod
    def extract_vgg_layers(layer_names):
        """Creates a vgg model that returns a list of intermediate
        output values. This way you can let it output the values of
        the intermediate convolutional layers.
        It creates a keras Model with a single input and multiple
        outputs depending on the number of layer_names.

        Args:
            layer_names (list[str]): the exact names to extract from the vgg network.

        Returns:
            tf.keras.Model
        """
        # Load our model. Load pretrained VGG, trained on imagenet data
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False

        outputs = [vgg.get_layer(name).output for name in layer_names]

        model = tf.keras.Model([vgg.input], outputs)
        return model
