# ported from spektral

from keras import ops
from keras.layers import GRUCell

from k3_node.layers.conv.message_passing import MessagePassing


class GatedGraphConv(MessagePassing):
    """
    `k3_node.layers.GatedGraphConv` 

    Implementation of Gated Graph Convolution (GGC) layer

    Args:
        channels: The number of output channels.
        n_layers: The number of GGC layers to stack.
        activation: Activation function to use.
        use_bias: Whether to add a bias to the linear transformation.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer for the `kernel` weights matrix.
        bias_regularizer: Regularizer for the bias vector.
        activity_regularizer: Regularizer for the output.
        kernel_constraint: Constraint for the `kernel` weights matrix.
        bias_constraint: Constraint for the bias vector.
        **kwargs: Additional arguments to pass to the `MessagePassing` superclass.
    
    """
    def __init__(
        self,
        channels=None,
        n_layers=None,
        out_channels=None,
        num_layers=None,
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
        channels = out_channels if out_channels is not None else channels
        n_layers = num_layers if num_layers is not None else n_layers
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
        self.out_channels = channels
        self.n_layers = n_layers
        self.num_layers = n_layers

    def build(self, input_shape):
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

    def call(self, x, edge_index=None, edge_weight=None, **kwargs):
        is_legacy = False
        if edge_index is None and isinstance(x, (tuple, list)):
            x, a, _ = self.get_inputs(x)
            edge_index = a
            is_legacy = True

        F = ops.shape(x)[-1]
        if F < self.channels:
            to_pad = self.channels - F
            ndims = len(ops.shape(x)) - 1
            output = ops.pad(x, [[0, 0]] * ndims + [[0, to_pad]])
        else:
            output = x

        for i in range(self.n_layers):
            m = ops.matmul(output, self.kernel[i])
            if is_legacy:
                m = self.propagate(m, edge_index)
            else:
                m = self.propagate(edge_index, x=m, edge_weight=edge_weight)
            output = self.rnn(m, [output])[0]

        if hasattr(self, "activation") and self.activation is not None and callable(self.activation):
            output = self.activation(output)
        return output

    @property
    def config(self):
        return {
            "channels": self.channels,
            "n_layers": self.n_layers,
        }
