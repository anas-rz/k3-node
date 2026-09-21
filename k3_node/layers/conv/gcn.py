# ported from stellargraph
from keras import ops
from keras import activations, initializers, constraints, regularizers
from keras.layers import Layer, dot


class GraphConvolution(Layer):
    """
    `k3_node.layers.GraphConvolution` 
    Implementation of Graph Convolution (GCN) layer

    Args:
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use.
        use_bias: Whether to add a bias to the linear transformation.
        final_layer: Deprecated, use tf.gather or GatherIndices instead.
        input_dim: Deprecated, use `keras.layers.Input` with `input_shape` instead.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        kernel_regularizer: Regularizer for the `kernel` weights matrix.
        kernel_constraint: Constraint for the `kernel` weights matrix.
        bias_initializer: Initializer for the bias vector.
        bias_regularizer: Regularizer for the bias vector.
        bias_constraint: Constraint for the bias vector.
        **kwargs: Additional arguments to pass to the `Layer` superclass.
    """
    def __init__(
        self,
        units,
        activation=None,
        use_bias=True,
        final_layer=None,
        input_dim=None,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        kernel_constraint=None,
        bias_initializer="zeros",
        bias_regularizer=None,
        bias_constraint=None,
        **kwargs,
    ):
        if isinstance(activation, int):
            # Called as GraphConvolution(in_channels, out_channels)
            self.in_channels = units
            units = activation
            activation = kwargs.pop("activation", None)

        if "input_shape" not in kwargs and input_dim is not None:
            kwargs["input_shape"] = (input_dim,)

        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        if final_layer is not None:
            raise ValueError(
                "'final_layer' is not longer supported, use 'tf.gather' or 'GatherIndices' separately"
            )

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_constraint = constraints.get(bias_constraint)

        super().__init__(**kwargs)

    def build(self, input_shapes):
        if isinstance(input_shapes, (list, tuple)) and len(input_shapes) > 0 and isinstance(input_shapes[0], (list, tuple)):
            feat_shape = input_shapes[0]
        else:
            feat_shape = input_shapes
        input_dim = int(feat_shape[-1]) if feat_shape is not None and feat_shape[-1] is not None else 8

        self.kernel = self.add_weight(
            shape=(1, input_dim, self.units),
            initializer=self.kernel_initializer,
            name="kernel",
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units,),
                initializer=self.bias_initializer,
                name="bias",
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.bias = None
        self.built = True

    def call(self, inputs, A=None, **kwargs):
        if A is not None:
            features = inputs
        elif isinstance(inputs, (list, tuple)):
            features, A = inputs
        else:
            features, A = inputs, None

        if A is not None and hasattr(A, "shape") and len(A.shape) == 2 and A.shape[0] == 2 and A.shape[1] != 2:
            num_nodes = ops.shape(features)[-2]
            a_dense = ops.zeros((num_nodes, num_nodes), dtype=features.dtype)
            indices = ops.transpose(A, axes=[1, 0])
            updates = ops.ones(shape=(ops.shape(A)[1],), dtype=features.dtype)
            A = ops.scatter_update(a_dense, indices, updates)

        was_2d = len(ops.shape(features)) == 2
        if was_2d:
            features = ops.expand_dims(features, 0)
        if len(ops.shape(A)) == 2:
            A = ops.expand_dims(A, 0)

        # Calculate the layer operation of GCN

        h_graph = dot((A, features), axes=1)
        b = ops.shape(h_graph)[0]
        kernel = ops.repeat(self.kernel, b, axis=0)
        output = dot((h_graph, kernel), axes=(-1, 1))

        # Add optional bias & apply activation
        if self.bias is not None:
            output += self.bias
        output = self.activation(output)

        if was_2d:
            output = ops.squeeze(output, 0)

        return output
