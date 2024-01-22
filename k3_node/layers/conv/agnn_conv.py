# ported from spektral
import keras
from keras import ops

from k3_node.layers.conv.message_passing import MessagePassing


def segment_softmax(x, indices, n_nodes=None):
    n_nodes = ops.max(indices) + 1 if n_nodes is None else n_nodes
    e_x = ops.exp(x - ops.take(ops.segment_max(x, indices, n_nodes), indices))
    e_x /= ops.take(ops.segment_sum(e_x, indices, n_nodes) + 1e-9, indices)
    return e_x


class AGNNConv(MessagePassing):
    def __init__(self, trainable=True, aggregate="sum", activation=None, **kwargs):
        super().__init__(aggregate=aggregate, activation=activation, **kwargs)
        self.trainable = trainable

    def build(self, input_shape):
        assert len(input_shape) >= 2
        if self.trainable:
            self.beta = self.add_weight(shape=(1,), initializer="ones", name="beta")
        else:
            self.beta = ops.cast(1.0, self.dtype)
        self.built = True

    def call(self, inputs, **kwargs):
        x, a, _ = self.get_inputs(inputs)
        x_norm = keras.utils.normalize(x, axis=-1)
        output = self.propagate(x, a, x_norm=x_norm)
        output = self.activation(output)

        return output

    def message(self, x, x_norm=None):
        x_j = self.get_sources(x)
        x_norm_i = self.get_targets(x_norm)
        x_norm_j = self.get_sources(x_norm)
        alpha = self.beta * ops.sum(x_norm_i * x_norm_j, axis=-1)

        if len(alpha.shape) == 2:
            alpha = ops.transpose(alpha)  # For mixed mode
        alpha = segment_softmax(alpha, self.index_targets, self.n_nodes)
        if len(alpha.shape) == 2:
            alpha = ops.transpose(alpha)  # For mixed mode
        alpha = alpha[..., None]

        return alpha * x_j

    @property
    def config(self):
        return {
            "trainable": self.trainable,
        }
