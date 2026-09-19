from typing import Union, Tuple
from keras import ops
from keras.layers import Dense

from k3_node.layers.conv.message_passing import MessagePassing


class GMMConv(MessagePassing):
    r"""The gaussian mixture model convolutional operator from the `"Geometric
    Deep Learning on Graphs and Manifolds using Mixture Model CNNs"
    <https://arxiv.org/abs/1611.08402>`_ paper.
    """
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        dim: int,
        kernel_size: int,
        separate_gaussians: bool = False,
        aggr: str = "mean",
        root_weight: bool = True,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim
        self.kernel_size = kernel_size
        self.separate_gaussians = separate_gaussians
        self.root_weight = root_weight
        self.use_bias = bias

        if isinstance(in_channels, int):
            self.in_channels_l = in_channels
            self.in_channels_r = in_channels
        else:
            self.in_channels_l, self.in_channels_r = in_channels

        self.g = self.add_weight(
            shape=(self.in_channels_l, out_channels * kernel_size),
            initializer="glorot_uniform",
            name="g",
        )

        if not separate_gaussians:
            self.mu = self.add_weight(
                shape=(kernel_size, dim),
                initializer="glorot_uniform",
                name="mu",
            )
            self.sigma = self.add_weight(
                shape=(kernel_size, dim),
                initializer="ones",
                name="sigma",
            )
        else:
            self.mu = self.add_weight(
                shape=(self.in_channels_l, out_channels, kernel_size, dim),
                initializer="glorot_uniform",
                name="mu",
            )
            self.sigma = self.add_weight(
                shape=(self.in_channels_l, out_channels, kernel_size, dim),
                initializer="ones",
                name="sigma",
            )

        if root_weight:
            self.root = Dense(out_channels, use_bias=False)
        else:
            self.root = None

        if bias:
            self.bias = self.add_weight(
                shape=(out_channels,),
                initializer="zeros",
                name="bias",
            )
        else:
            self.bias = None

    def build(self, input_shape=None):
        if self.root is not None:
            self.root.build((None, self.in_channels_r))
        self.built = True

    def call(self, inputs, edge_index=None, edge_attr=None, **kwargs):
        if edge_index is None:
            if isinstance(inputs, (list, tuple)):
                if len(inputs) == 3:
                    x, edge_index, edge_attr = inputs
                elif len(inputs) == 2:
                    x, edge_index = inputs
                else:
                    raise ValueError(f"Unexpected input length {len(inputs)}")
            else:
                raise ValueError("Expected (x, edge_index) or x and edge_index")
        else:
            x = inputs

        if not self.built:
            self.build()

        if isinstance(x, (list, tuple)):
            x_src, x_dst = x
        else:
            x_src = x_dst = x

        num_nodes = ops.shape(x_dst)[0]

        if not self.separate_gaussians:
            x_l = ops.matmul(x_src, self.g)
            out = self.propagate(
                edge_index,
                x=(x_l, x_dst),
                edge_attr=edge_attr,
                size=(ops.shape(x_src)[0], num_nodes),
            )
        else:
            out = self.propagate(
                edge_index,
                x=(x_src, x_dst),
                edge_attr=edge_attr,
                size=(ops.shape(x_src)[0], num_nodes),
            )

        if self.root is not None:
            out = out + self.root(x_dst)

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j, edge_attr):
        EPS = 1e-15
        M = self.out_channels
        K = self.kernel_size

        if not self.separate_gaussians:
            # edge_attr: (E, D), mu: (K, D), sigma: (K, D)
            diff = ops.expand_dims(edge_attr, 1) - ops.expand_dims(self.mu, 0)
            gaussian = -0.5 * ops.square(diff) / (EPS + ops.square(ops.expand_dims(self.sigma, 0)))
            gaussian = ops.exp(ops.sum(gaussian, axis=-1))  # (E, K)

            x_j_reshaped = ops.reshape(x_j, (-1, K, M))
            return ops.sum(x_j_reshaped * ops.expand_dims(gaussian, -1), axis=1)
        else:
            F = self.in_channels_l
            diff = ops.expand_dims(ops.expand_dims(ops.expand_dims(edge_attr, 1), 1), 1) - ops.expand_dims(self.mu, 0)
            gaussian = -0.5 * ops.square(diff) / (EPS + ops.square(ops.expand_dims(self.sigma, 0)))
            gaussian = ops.exp(ops.sum(gaussian, axis=-1))  # (E, F, M, K)
            g_reshaped = ops.reshape(self.g, (1, F, M, K))
            gaussian = ops.sum(gaussian * g_reshaped, axis=-1)  # (E, F, M)
            return ops.sum(ops.expand_dims(x_j, -1) * gaussian, axis=1)

