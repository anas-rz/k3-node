# ported from spektral

from keras import ops
from keras.layers import Dense

from k3_node.layers.conv.message_passing import MessagePassing


class CrystalConv(MessagePassing):
    """
    `k3_node.layers.CrystalConv`
    Implementation of Crystal Graph Convolutional Neural Networks (CGCNN) layer

    Args:
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
        channels = input_shape[0][-1]
        self.dense_f = Dense(channels, activation="sigmoid", **layer_kwargs)
        self.dense_s = Dense(channels, activation=self.activation, **layer_kwargs)

        self.built = True

    def message(self, x, e=None):
        x_i = self.get_targets(x)
        x_j = self.get_sources(x)

        to_concat = [x_i, x_j]
        if e is not None:
            to_concat += [e]
        z = ops.concatenate(to_concat, axis=-1)
        output = self.dense_s(z) * self.dense_f(z)

        return output

    def update(self, embeddings, x=None):
        return x + embeddings
