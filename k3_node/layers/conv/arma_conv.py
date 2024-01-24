# ported from spektral

from keras import activations
from keras import ops
from keras.layers import Dropout

from k3_node.layers.conv.conv import Conv
from k3_node.ops import modal_dot, normalized_adjacency


class ARMAConv(Conv):
    def __init__(
        self,
        channels,
        order=1,
        iterations=1,
        share_weights=False,
        gcn_activation="relu",
        dropout_rate=0.0,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        super().__init__(
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs,
        )
        self.channels = channels
        self.iterations = iterations
        self.order = order
        self.share_weights = share_weights
        self.gcn_activation = activations.get(gcn_activation)
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        assert len(input_shape) >= 2
        F = input_shape[0][-1]

        # Create weights for parallel stacks
        # self.kernels[k][i] refers to the k-th stack, i-th iteration
        self.kernels = []
        for k in range(self.order):
            kernel_stack = []
            current_shape = F
            for i in range(self.iterations):
                kernel_stack.append(
                    self.create_weights(
                        current_shape, F, self.channels, "ARMA_GCS_{}{}".format(k, i)
                    )
                )
                current_shape = self.channels
                if self.share_weights and i == 1:
                    # No need to continue because all weights will be shared
                    break
            self.kernels.append(kernel_stack)

        self.dropout = Dropout(self.dropout_rate, dtype=self.dtype)
        self.built = True

    def call(self, inputs, mask=None):
        x, a = inputs

        output = []
        for k in range(self.order):
            output_k = x
            for i in range(self.iterations):
                output_k = self.gcs([output_k, x, a], k, i)
            output.append(output_k)
        output = ops.stack(output, axis=-1)
        output = ops.mean(output, axis=-1)

        if mask[0] is not None:
            output *= mask[0]
        output = self.activation(output)

        return output

    def create_weights(self, input_dim, input_dim_skip, channels, name):
        kernel_1 = self.add_weight(
            shape=(input_dim, channels),
            name=name + "_kernel_1",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        kernel_2 = self.add_weight(
            shape=(input_dim_skip, channels),
            name=name + "_kernel_2",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        bias = None
        if self.use_bias:
            bias = self.add_weight(
                shape=(channels,),
                name=name + "_bias",
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        return kernel_1, kernel_2, bias

    def gcs(self, inputs, stack, iteration):
        x, x_skip, a = inputs

        itr = 1 if self.share_weights and iteration >= 1 else iteration
        kernel_1, kernel_2, bias = self.kernels[stack][itr]

        output = ops.dot(x, kernel_1)
        output = modal_dot(a, output)

        skip = ops.dot(x_skip, kernel_2)
        skip = self.dropout(skip)
        output += skip

        if self.use_bias:
            output = ops.add(output, bias)
        output = self.gcn_activation(output)

        return output

    @property
    def config(self):
        return {
            "channels": self.channels,
            "iterations": self.iterations,
            "order": self.order,
            "share_weights": self.share_weights,
            "gcn_activation": activations.serialize(self.gcn_activation),
            "dropout_rate": self.dropout_rate,
        }

    @staticmethod
    def preprocess(a):
        return normalized_adjacency(a, symmetric=True)
