from typing import Union, Tuple
from keras import layers, ops

from k3_node.layers.conv.message_passing import MessagePassing


class LEConv(MessagePassing):
    r"""The local extremum graph convolutional operator from the
    `"ASAP: Adaptive Structure Aware Pooling for Learning Hierarchical Graph
    Representations" <https://arxiv.org/abs/1911.07979>`_ paper.

    Args:
        in_channels: Size of each input sample, or a tuple for bipartite graphs.
        out_channels: Size of each output sample.
        bias: If set to :obj:`False`, the layer will not learn an additive bias.
            (default: ``True``)
    """

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(aggr="add", **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_bias = bias

        self.lin1 = layers.Dense(out_channels, use_bias=bias)
        self.lin2 = layers.Dense(out_channels, use_bias=False)
        self.lin3 = layers.Dense(out_channels, use_bias=bias)

    def build(self, input_shape):
        if isinstance(input_shape, (tuple, list)) and len(input_shape) > 0 and isinstance(input_shape[0], (tuple, list)):
            in_channels_src = input_shape[0][-1]
            in_channels_dst = input_shape[1][-1] if len(input_shape) > 1 and input_shape[1] is not None else in_channels_src
        else:
            in_channels_src = input_shape[-1]
            in_channels_dst = input_shape[-1]

        self.lin1.build((None, in_channels_src))
        self.lin2.build((None, in_channels_dst))
        self.lin3.build((None, in_channels_dst))
        self.built = True

    def call(self, x, edge_index=None, edge_weight=None, **kwargs):
        if edge_index is None and isinstance(x, (tuple, list)):
            x, edge_index = x[0], x[1]

        if not isinstance(x, (tuple, list)):
            x_src, x_dst = x, x
        else:
            x_src, x_dst = x[0], x[1]

        a = self.lin1(x_src)
        b = self.lin2(x_dst)

        out = self.propagate(edge_index, a=a, b=b, edge_weight=edge_weight)
        return out + self.lin3(x_dst)

    def message(self, a_j, b_i, edge_weight=None):
        out = a_j - b_i
        if edge_weight is None:
            return out
        return out * ops.expand_dims(edge_weight, -1)

