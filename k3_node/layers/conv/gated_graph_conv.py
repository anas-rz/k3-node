# ported from spektral

from keras import ops
from keras.layers import GRUCell

from k3_node.layers.conv.message_passing import MessagePassing


class GatedGraphConv(MessagePassing):
    def __init__(
        self,
        channels,
        n_layers,
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
        self.n_layers = n_layers

    def build(self, input_shape):
        assert len(input_shape) >= 2
        self.kernel = self.add_weight(
            name="kernel",
            shape=(self.n_layers, self.channels, self.channels),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.rnn = GRUCell(
            self.channels,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            activity_regularizer=self.activity_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
            use_bias=self.use_bias,
            dtype=self.dtype,
        )
        self.built = True

    def call(self, inputs):
        x, a, _ = self.get_inputs(inputs)
        F = ops.shape(x)[-1]

        to_pad = self.channels - F
        ndims = len(x.shape) - 1
        output = ops.pad(x, [[0, 0]] * ndims + [[0, to_pad]])
        for i in range(self.n_layers):
            m = ops.matmul(output, self.kernel[i])
            m = self.propagate(m, a)
            output = self.rnn(m, [output])[0]

        output = self.activation(output)
        return output

    @property
    def config(self):
        return {
            "channels": self.channels,
            "n_layers": self.n_layers,
        }
