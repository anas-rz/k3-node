# ported from spektral

from keras import ops
from keras import activations
from keras.layers import BatchNormalization, Dense
from keras.models import Sequential

from k3_node.layers.conv.message_passing import MessagePassing


class GINConv(MessagePassing):
    """
    `k3_node.layers.GINConv` 
    Implementation of Graph Isomorphism Network (GIN) layer

    Args:
        channels: The number of output channels.
        epsilon: The epsilon parameter for the MLP.
        mlp_hidden: A list of hidden channels for the MLP.
        mlp_activation: The activation function to use in the MLP.
        mlp_batchnorm: Whether to use batch normalization in the MLP.
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
        channels,
        epsilon=None,
        mlp_hidden=None,
        mlp_activation="relu",
        mlp_batchnorm=True,
        aggregate="sum",
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
            aggregate=aggregate,
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
        self.epsilon = epsilon
        self.mlp_hidden = mlp_hidden if mlp_hidden else []
        self.mlp_activation = activations.get(mlp_activation)
        self.mlp_batchnorm = mlp_batchnorm

    def build(self, input_shape):
        assert len(input_shape) >= 2
        layer_kwargs = dict(
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
        )

        self.mlp = Sequential()
        for channels in self.mlp_hidden:
            self.mlp.add(Dense(channels, self.mlp_activation, **layer_kwargs))
            if self.mlp_batchnorm:
                self.mlp.add(BatchNormalization())
        self.mlp.add(
            Dense(
                self.channels, self.activation, use_bias=self.use_bias, **layer_kwargs
            )
        )

        if self.epsilon is None:
            self.eps = self.add_weight(shape=(1,), initializer="zeros", name="eps")
        else:
            # If epsilon is given, keep it constant
            self.eps = ops.cast(self.epsilon, self.dtype)
        self.one = ops.cast(1, self.dtype)

        self.built = True

    def call(self, inputs, **kwargs):
        x, a, _ = self.get_inputs(inputs)
        output = self.mlp((self.one + self.eps) * x + self.propagate(x, a))

        return output

    @property
    def config(self):
        return {
            "channels": self.channels,
            "epsilon": self.epsilon,
            "mlp_hidden": self.mlp_hidden,
            "mlp_activation": self.mlp_activation,
            "mlp_batchnorm": self.mlp_batchnorm,
        }
