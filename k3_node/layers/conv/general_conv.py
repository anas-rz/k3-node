# ported from spektral

from keras import ops
from keras import activations
from keras.layers import BatchNormalization, Dropout, PReLU

from k3_node.layers.conv.message_passing import MessagePassing


class GeneralConv(MessagePassing):
    """
    `k3_node.layers.GeneralConv`
    Implementation of General Graph Convolution

    Args:
        channels: The number of output channels.
        batch_norm: Whether to use batch normalization.
        dropout: The dropout rate.
        aggregate: Aggregation function to use (one of 'sum', 'mean', 'max').
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
        channels=256,
        batch_norm=True,
        dropout=0.0,
        aggregate="sum",
        activation="prelu",
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
            aggregate=aggregate,
            activation=None,
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
        self.dropout_rate = dropout
        self.use_batch_norm = batch_norm
        if activation == "prelu" or "prelu" in kwargs:
            self.activation = PReLU()
        else:
            self.activation = activations.get(activation)

    def build(self, input_shape):
        input_dim = input_shape[0][-1]
        self.dropout = Dropout(self.dropout_rate)
        if self.use_batch_norm:
            self.batch_norm = BatchNormalization()
        self.kernel = self.add_weight(
            shape=(input_dim, self.channels),
            initializer=self.kernel_initializer,
            name="kernel",
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.channels,),
                initializer=self.bias_initializer,
                name="bias",
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        self.built = True

    def call(self, inputs, **kwargs):
        x, a, _ = self.get_inputs(inputs)

        # TODO: a = add_self_loops(a)

        x = ops.matmul(x, self.kernel)
        if self.use_bias:
            x = ops.add(x, self.bias)
        if self.use_batch_norm:
            x = self.batch_norm(x)
        x = self.dropout(x)
        x = self.activation(x)

        return self.propagate(x, a)

    @property
    def config(self):
        config = {
            "channels": self.channels,
        }
        if self.activation.__class__.__name__ == "PReLU":
            config["prelu"] = True

        return config
