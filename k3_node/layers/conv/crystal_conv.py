# ported from spektral

from keras import ops
from keras.layers import Dense

from k3_node.layers.conv.message_passing import MessagePassing


class CrystalConv(MessagePassing):
    """
    `k3_node.layers.CrystalConv`
    Implementation of Crystal Graph Convolutional Neural Networks (CGCNN) layer

    Args:
        channels: The number of channels/units (optional).
        edge_dim: The dimensionality of edge features (optional).
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
        channels=None,
        edge_dim=None,
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
        self.edge_dim = edge_dim

    def build(self, input_shape=None):
        layer_kwargs = dict(
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
            dtype=self.dtype,
        )
        if self.channels is not None:
            channels = self.channels
        elif input_shape is not None:
            if isinstance(input_shape, (list, tuple)) and len(input_shape) > 0 and isinstance(input_shape[0], (list, tuple)):
                channels = input_shape[0][-1]
            elif isinstance(input_shape, (list, tuple)) and len(input_shape) > 0 and isinstance(input_shape[-1], int):
                channels = input_shape[-1]
            else:
                channels = 16
        else:
            channels = 16

        self.channels = channels
        self.dense_f = Dense(channels, activation="sigmoid", **layer_kwargs)
        self.dense_s = Dense(channels, activation=self.activation, **layer_kwargs)

        self.built = True

    def call(self, x, edge_index=None, edge_attr=None, **kwargs):
        if not self.built:
            x_shape = getattr(x, "shape", None)
            self.build(x_shape)

        if edge_index is None and isinstance(x, (tuple, list)):
            x_in, a, e = self.get_inputs(x)
            return self.propagate(x_in, a, e, **kwargs)

        if edge_attr is None:
            edge_attr = kwargs.get("e", None)

        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i=None, x_j=None, edge_attr=None, x=None, e=None, **kwargs):
        if x_i is None and x is not None:
            x_i = self.get_targets(x)
            x_j = self.get_sources(x)
        if edge_attr is None:
            edge_attr = e

        to_concat = [x_i, x_j]
        if e is not None:
            to_concat += [e]
        if edge_attr is not None:
            to_concat.append(edge_attr)
        z = ops.concatenate(to_concat, axis=-1)
        output = self.dense_s(z) * self.dense_f(z)

        return output

    def update(self, embeddings, x=None, **kwargs):
        if x is None:
            return embeddings
        if isinstance(x, (tuple, list)):
            x = x[1] if x[1] is not None else x[0]
        return x + embeddings
