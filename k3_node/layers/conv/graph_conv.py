from typing import Optional, Union, Tuple
import keras
from keras import layers, ops

from k3_node.layers.conv.message_passing import MessagePassing


class GraphConv(MessagePassing):
    r"""The graph neural network operator from the `"Weisfeiler and Leman Go
    Neural: Higher-order Graph Neural Networks"
    <https://arxiv.org/abs/1810.02244>`_ paper.

    Args:
        in_channels: Size of each input sample, or a tuple for bipartite graphs.
        out_channels: Size of each output sample.
        aggr: The aggregation scheme to use (``"add"``, ``"mean"``, ``"max"``).
            (default: ``"add"``)
        bias: If set to :obj:`False`, the layer will not learn
            an additive bias. (default: ``"True"``)
    """

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        aggr: str = "add",
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(aggr=aggr, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_bias = bias

        self.lin_rel = layers.Dense(out_channels, use_bias=bias)
        self.lin_root = layers.Dense(out_channels, use_bias=False)

    def build(self, input_shape):
        if isinstance(input_shape, (tuple, list)) and len(input_shape) > 0 and isinstance(input_shape[0], (tuple, list)):
            in_channels_l = input_shape[0][-1]
            in_channels_r = input_shape[1][-1] if len(input_shape) > 1 and input_shape[1] is not None else in_channels_l
        else:
            in_channels_l = input_shape[-1]
            in_channels_r = input_shape[-1]

        self.lin_rel.build((None, in_channels_l))
        self.lin_root.build((None, in_channels_r))
        self.built = True

    def call(self, x, edge_index=None, edge_weight=None, size=None, **kwargs):
        if edge_index is None and isinstance(x, (tuple, list)):
            x, edge_index = x[0], x[1]

        if not isinstance(x, (tuple, list)):
            x_src = x
            x_dst = x
        else:
            x_src, x_dst = x[0], x[1]

        out = self.propagate(edge_index, x=(x_src, x_dst), edge_weight=edge_weight, size=size)
        out = self.lin_rel(out)
        if x_dst is not None:
            out = out + self.lin_root(x_dst)
        return out

    def message(self, x_j, edge_weight=None):
        if edge_weight is None:
            return x_j
        return ops.expand_dims(edge_weight, -1) * x_j

