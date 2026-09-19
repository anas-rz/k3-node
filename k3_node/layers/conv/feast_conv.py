from typing import Optional, Union, Tuple
import keras
from keras import ops
from keras.layers import Dense

from k3_node.layers.conv.message_passing import MessagePassing
from k3_node.layers.conv.utils import remove_self_loops, add_self_loops


class FeaStConv(MessagePassing):
    r"""The (fault-tolerant) feature-steered graph convolution operator from
    the `"FeaStNet: Feature-Steered Graph Convolutions for 3D Shape Analysis"
    <https://arxiv.org/abs/1706.05206>`_ paper.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        add_self_loops: bool = True,
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "mean")
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.add_self_loops = add_self_loops
        self.use_bias = bias

        self.lin = Dense(heads * out_channels, use_bias=False)
        self.u = self.add_weight(
            shape=(in_channels, heads),
            initializer="glorot_uniform",
            name="u",
        )
        self.c = self.add_weight(
            shape=(heads,),
            initializer="zeros",
            name="c",
        )

        if bias:
            self.bias = self.add_weight(
                shape=(out_channels,),
                initializer="zeros",
                name="bias",
            )
        else:
            self.bias = None

    def build(self, input_shape=None):
        self.lin.build((None, self.in_channels))
        self.built = True

    def call(self, inputs, edge_index=None, **kwargs):
        if edge_index is None:
            if isinstance(inputs, (list, tuple)) and len(inputs) == 2:
                x, edge_index = inputs
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
        if self.add_self_loops:
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

        out = self.propagate(
            edge_index,
            x=(x_src, x_dst),
            size=(ops.shape(x_src)[0], num_nodes),
        )

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_i, x_j):
        q = ops.matmul(x_j - x_i, self.u) + self.c
        q = ops.softmax(q, axis=-1)
        x_j_mapped = ops.reshape(self.lin(x_j), (-1, self.heads, self.out_channels))
        return ops.sum(x_j_mapped * ops.expand_dims(q, -1), axis=1)

