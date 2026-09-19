from typing import Union, Tuple
from keras import layers, ops

from k3_node.layers.conv.message_passing import MessagePassing


class CGConv(MessagePassing):
    r"""The Crystal Graph Convolutional operator from the
    `"Crystal Graph Convolutional Neural Networks for an Accurate and
    Interpretable Prediction of Material Properties"
    <https://arxiv.org/abs/1710.10324>`_ paper.

    Args:
        channels: Size of each input sample, or a tuple for bipartite graphs.
        dim: Edge feature dimensionality. (default: ``0``)
        aggr: The aggregation scheme to use (``"add"``, ``"mean"``, ``"max"``).
            (default: ``"add"``)
        batch_norm: If set to :obj:`True`, will apply batch normalization.
            (default: ``False``)
        bias: If set to :obj:`False`, the layer will not learn an additive bias.
            (default: ``True``)
    """

    def __init__(
        self,
        channels: Union[int, Tuple[int, int]],
        dim: int = 0,
        aggr: str = "add",
        batch_norm: bool = False,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(aggr=aggr, **kwargs)
        self.channels = channels
        self.dim = dim
        self.batch_norm = batch_norm
        self.use_bias = bias

        if isinstance(channels, int):
            self.in_channels_src = channels
            self.in_channels_dst = channels
        else:
            self.in_channels_src = channels[0]
            self.in_channels_dst = channels[1]

        in_dim = self.in_channels_src + self.in_channels_dst + dim
        self.lin_f = layers.Dense(self.in_channels_dst, use_bias=bias)
        self.lin_s = layers.Dense(self.in_channels_dst, use_bias=bias)
        self.bn = layers.BatchNormalization() if batch_norm else None

    def build(self, input_shape):
        in_dim = self.in_channels_src + self.in_channels_dst + self.dim
        self.lin_f.build((None, in_dim))
        self.lin_s.build((None, in_dim))
        self.built = True

    def call(self, x, edge_index=None, edge_attr=None, **kwargs):
        if edge_index is None and isinstance(x, (tuple, list)):
            x, edge_index = x[0], x[1]

        if not isinstance(x, (tuple, list)):
            x_src, x_dst = x, x
        else:
            x_src, x_dst = x[0], x[1]

        out = self.propagate(edge_index, x=(x_src, x_dst), edge_attr=edge_attr)
        if self.bn is not None:
            out = self.bn(out)
        out = x_dst + out
        return out

    def message(self, x_i, x_j, edge_attr=None):
        if edge_attr is None:
            z = ops.concatenate([x_i, x_j], axis=-1)
        else:
            z = ops.concatenate([x_i, x_j, edge_attr], axis=-1)
        return ops.sigmoid(self.lin_f(z)) * ops.softplus(self.lin_s(z))

