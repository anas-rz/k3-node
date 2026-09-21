from keras import layers, ops

from k3_node.layers.conv.conv import Conv
from k3_node.ops import normalized_adjacency, polyval


class DiffuseFeatures(layers.Layer):
    def __init__(
        self,
        num_diffusion_steps,
        kernel_initializer,
        kernel_regularizer,
        kernel_constraint,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.K = num_diffusion_steps
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint

    def build(self, input_shape):
        # Initializing the kernel vector (R^K) (theta in paper)
        self.kernel = self.add_weight(
            shape=(self.K,),
            name="kernel",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )

    def call(self, inputs):
        x, a = inputs

        diffusion_matrix = polyval(self.kernel, a)
        diffused_features = ops.matmul(diffusion_matrix, x)
        H = ops.sum(diffused_features, axis=-1)
        return ops.expand_dims(H, -1)


class DiffusionConv(Conv):
    """
    `k3_node.layers.DiffusionConv`
    Implementation of Diffusion Convolutional Neural Networks (DCNN) layer

    Args:
        channels: The number of output channels.
        K: The number of diffusion steps.
        activation: Activation function to use.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        kernel_regularizer: Regularizer for the `kernel` weights matrix.
        kernel_constraint: Constraint for the `kernel` weights matrix.
        **kwargs: Additional arguments to pass to the `Conv` superclass.
    """
    def __init__(
        self,
        channels,
        out_channels=None,
        K=6,
        activation="tanh",
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        kernel_constraint=None,
        **kwargs,
    ):
        if out_channels is not None:
            self.in_channels = channels
            channels = out_channels
        super().__init__(
            activation=activation,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            kernel_constraint=kernel_constraint,
            **kwargs,
        )

        self.channels = channels
        self.K = K + 1

    def build(self, input_shape):
        self.filters = [
            DiffuseFeatures(
                num_diffusion_steps=self.K,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                kernel_constraint=self.kernel_constraint,
            )
            for _ in range(self.channels)
        ]
        for f in self.filters:
            f.build(None)
        super().build(input_shape)

    def apply_filters(self, x, a):
        diffused_features = []

        for diffusion in self.filters:
            diffused_feature = diffusion((x, a))
            diffused_features.append(diffused_feature)

        return ops.concatenate(diffused_features, -1)

    def call(self, inputs, a=None, **kwargs):
        if a is not None:
            x = inputs
        elif isinstance(inputs, (list, tuple)):
            x, a = inputs
        else:
            x, a = inputs, None

        if a is not None and hasattr(a, "shape") and len(a.shape) == 2 and a.shape[0] == 2 and a.shape[1] != 2:
            num_nodes = ops.shape(x)[0]
            a_dense = ops.zeros((num_nodes, num_nodes), dtype=x.dtype)
            indices = ops.transpose(a, axes=[1, 0])
            updates = ops.ones(shape=(ops.shape(a)[1],), dtype=x.dtype)
            a = ops.scatter_update(a_dense, indices, updates)

        output = self.apply_filters(x, a)

        output = self.activation(output)

        return output

    @property
    def config(self):
        return {"channels": self.channels, "K": self.K - 1}

    @staticmethod
    def preprocess(a):
        return normalized_adjacency(a)
