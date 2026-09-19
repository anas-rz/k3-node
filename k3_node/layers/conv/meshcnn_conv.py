from typing import Optional, List
import keras
from keras import ops
from k3_node.layers.conv.message_passing import MessagePassing


class MeshCNNConv(MessagePassing):
    r"""The MeshCNN convolutional operator from the `"MeshCNN: A Network With An Edge"
    <https://arxiv.org/abs/1809.05910>`_ paper.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        kernels (List[keras.layers.Layer], optional): A list of 5 neural network layers
            that transform edge representations. (default: :obj:`None`)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernels: Optional[List[keras.layers.Layer]] = None,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        if kernels is None:
            self.kernels = [
                keras.layers.Dense(out_channels, use_bias=True)
                for _ in range(5)
            ]
        else:
            assert len(kernels) == 5, "kernels must be a list of 5 layers"
            self.kernels = kernels

    def build(self, input_shape=None):
        for k in self.kernels:
            if hasattr(k, "build") and not k.built:
                k.build((None, self.in_channels))
        super().build(input_shape)

    def call(self, x, edge_index, **kwargs):
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        n_a = x_j[0::4]
        n_b = x_j[1::4]
        n_c = x_j[2::4]
        n_d = x_j[3::4]

        m1 = self.kernels[1](ops.abs(n_a - n_c))
        m2 = self.kernels[2](n_a + n_c)
        m3 = self.kernels[3](ops.abs(n_b - n_d))
        m4 = self.kernels[4](n_b + n_d)

        return ops.reshape(
            ops.stack([m1, m2, m3, m4], axis=1),
            (-1, self.out_channels),
        )

    def update(self, inputs, x):
        return self.kernels[0](x) + inputs

