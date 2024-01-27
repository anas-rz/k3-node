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

        diffusion_matrix = polyval(ops.unstack(self.kernel), a)
        diffused_features = ops.matmul(diffusion_matrix, x)
        H = ops.sum(diffused_features, axis=-1)
        return ops.expand_dims(H, -1)


class DiffusionConv(Conv):
    def __init__(
        self,
        channels,
        K=6,
        activation="tanh",
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        kernel_constraint=None,
        **kwargs,
    ):
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

    def apply_filters(self, x, a):
        diffused_features = []

        for diffusion in self.filters:
            diffused_feature = diffusion((x, a))
            diffused_features.append(diffused_feature)

        return ops.concatenate(diffused_features, -1)

    def call(self, inputs):
        x, a = inputs
        output = self.apply_filters(x, a)

        output = self.activation(output)

        return output

    @property
    def config(self):
        return {"channels": self.channels, "K": self.K - 1}

    @staticmethod
    def preprocess(a):
        return normalized_adjacency(a)