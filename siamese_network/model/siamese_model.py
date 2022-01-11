import tensorflow as tf
from tensorflow.keras import metrics
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from siamese_network.model.embedding import create_embedding_model
from siamese_network.constants import IMG_SPATIAL_SIZE


def l2_distance(emb1, emb2):
    return tf.reduce_sum(tf.square(emb1 - emb2), axis=-1)


class DistanceLayer(layers.Layer):
    """Custom layer that computes the distance between the anchor embedding
    and the positive embedding, and the anchor embedding and the negative
    embedding.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        """Calculates the distances between the anchor and positive, and anchor
        and negative images and returns the distances per sample.

        Args:
            anchor (tf.Tensor):
            positive (tf.Tensor):
            negative (tf.Tensor):

        Returns:
            tuple[tf.Tensor, tf.Tensor]
        """
        anchor_to_pos_distance = l2_distance(anchor, positive)
        anchor_to_neg_distance = l2_distance(anchor, negative)
        return anchor_to_pos_distance, anchor_to_neg_distance


def create_siamese_model(model_embedding):
    """Creates a siamese network that can be used to produce distances.

    Args:
        model_embedding (tf.keras.models.Model):

    Returns:
        tf.keras.models.Model
    """
    input_anchor = layers.Input(name="anchor", shape=IMG_SPATIAL_SIZE + (3,))
    input_positive = layers.Input(name="positive", shape=IMG_SPATIAL_SIZE + (3,))
    input_negative = layers.Input(name="negative", shape=IMG_SPATIAL_SIZE + (3,))

    distances = DistanceLayer()(
        model_embedding(input_anchor),
        model_embedding(input_positive),
        model_embedding(input_negative),
    )

    model_siamese = Model(inputs=[input_anchor, input_positive, input_negative], outputs=distances)

    return model_siamese


class SiameseModel(Model):
    """The Siamese Network model with a custom training and testing methods.

    Computes the triplet loss using the three embeddings produced by the
    Siamese Network.

    The triplet loss is defined as:

        L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0)

    where the inputs respectively are embeddings of the anchor, the positive
    and the negative images.
    """

    def __init__(self, margin=0.5):
        super(SiameseModel, self).__init__()
        self.embedding = create_embedding_model()
        self.siamese_network = create_siamese_model(self.embedding)
        self.margin = margin
        self.mean_triplet_loss = metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        # GradientTape is a context manager that records every operation that
        # you do inside. We are using it here to compute the loss so we can get
        # the gradients and apply them using the optimizer specified in `compile()`
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        # Storing the gradients of the loss function with respect to the
        # weights/parameters.
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        # Let's update and return the training loss metric.
        self.mean_triplet_loss.update_state(loss)
        return {"loss": self.mean_triplet_loss.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)

        # Let's update and return the loss metric.
        self.mean_triplet_loss.update_state(loss)
        return {"loss": self.mean_triplet_loss.result()}

    def _compute_loss(self, data):
        # The output of the network is a tuple containing the distances
        # between the anchor and the positive example, and the anchor and
        # the negative example.
        ap_distance, an_distance = self.siamese_network(data)

        # Computing the Triplet Loss by subtracting both distances and
        # making sure we don't get a negative value.
        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.mean_triplet_loss]
