# ported from spektral

from keras import activations
from keras.layers import Dense, Dropout
from keras.models import Sequential

from k3_node.layers.conv.conv import Conv
from k3_node.ops import gcn_filter, modal_dot


class APPNPConv(Conv):
    """
        `k3_node.layers.APPNPConv`
        Implementation of Approximate Personalized Propagation of Neural Predictions

        Args:
            channels: The number of output channels.
            alpha: The teleport probability.
            propagations: The number of propagation steps.
            mlp_hidden: A list of hidden channels for the MLP.
            mlp_activation: The activation function to use in the MLP.
            dropout_rate: The dropout rate for the MLP.
            activation: The activation function to use in the layer.
            use_bias: Whether to add a bias to the linear transformation.
            kernel_initializer: Initializer for the `kernel` weights matrix.
            bias_initializer: Initializer for the bias vector.
            kernel_regularizer: Regularizer for the `kernel` weights matrix.
            bias_regularizer: Regularizer for the bias vector.
            activity_regularizer: Regularizer for the output.
            kernel_constraint: Constraint for the `kernel` weights matrix.
            bias_constraint: Constraint for the bias vector.
            **kwargs: Additional keyword arguments.
    """
    def __init__(
        self,
        channels,
        alpha=0.2,
        propagations=1,
        mlp_hidden=None,
        mlp_activation="relu",
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
        self.mlp_hidden = mlp_hidden if mlp_hidden else []
        self.alpha = alpha
        self.propagations = propagations
        self.mlp_activation = activations.get(mlp_activation)
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        assert len(input_shape) >= 2
        layer_kwargs = dict(
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
            dtype=self.dtype,
        )
        mlp_layers = []
        for channels in self.mlp_hidden:
            mlp_layers.extend(
                [
                    Dropout(self.dropout_rate),
                    Dense(channels, self.mlp_activation, **layer_kwargs),
                ]
            )
        mlp_layers.append(Dense(self.channels, "linear", **layer_kwargs))
        self.mlp = Sequential(mlp_layers)
        self.built = True

    def call(self, inputs, mask=None):
        x, a = inputs
        mlp_out = self.mlp(x)
        output = mlp_out
        for _ in range(self.propagations):
            output = (1 - self.alpha) * modal_dot(a, output) + self.alpha * mlp_out
        if mask[0] is not None:
            output *= mask[0]
        output = self.activation(output)

        return output

    @property
    def config(self):
        return {
            "channels": self.channels,
            "alpha": self.alpha,
            "propagations": self.propagations,
            "mlp_hidden": self.mlp_hidden,
            "mlp_activation": activations.serialize(self.mlp_activation),
            "dropout_rate": self.dropout_rate,
        }

    @staticmethod
    def preprocess(a):
        return gcn_filter(a)
