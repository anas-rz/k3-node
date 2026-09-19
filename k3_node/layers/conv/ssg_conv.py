from typing import Optional, Union, Tuple
import keras
from keras import layers, ops

from k3_node.layers.conv.message_passing import MessagePassing
from k3_node.layers.conv.utils import gcn_norm


class SSGConv(MessagePassing):
    r"""The simple spectral graph convolutional operator from the
    `"Simple Spectral Graph Convolution" <https://arxiv.org/abs/2109.07191>`_ paper.

    Args:
        in_channels: Size of each input sample.
        out_channels: Size of each output sample.
        alpha: Teleport probability :math:`\alpha`.
        K: Number of hops :math:`K`. (default: ``1``)
        cached: If set to :obj:`True`, the layer will cache normalization coefficients.
            (default: ``False``)
        add_self_loops: If set to :obj:`False`, will not add self-loops.
            (default: ``True``)
        bias: If set to :obj:`False`, the layer will not learn an additive bias.
            (default: ``True``)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        alpha: float,
        K: int = 1,
        cached: bool = False,
        add_self_loops: bool = True,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(aggr="add", **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.alpha = alpha
        self.K = K
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.use_bias = bias

        self.lin = layers.Dense(out_channels, use_bias=bias)
        self._cached_edge_index = None
        self._cached_norm = None

    def build(self, input_shape):
        feat_shape = input_shape[0] if isinstance(input_shape, (tuple, list)) and isinstance(input_shape[0], (tuple, list)) else input_shape
        self.lin.build(feat_shape)
        self.built = True

    def call(self, x, edge_index=None, edge_weight=None, **kwargs):
        if edge_index is None and isinstance(x, (tuple, list)):
            x, edge_index = x[0], x[1]

        if self.cached and self._cached_edge_index is not None:
            edge_index = self._cached_edge_index
            edge_weight = self._cached_norm
        else:
            num_nodes = int(ops.shape(x)[self.node_dim])
            edge_index, edge_weight = gcn_norm(
                edge_index,
                edge_weight,
                num_nodes=num_nodes,
                add_self_loops=self.add_self_loops,
                flow=self.flow,
                dtype=x.dtype,
            )
            if self.cached:
                self._cached_edge_index = edge_index
                self._cached_norm = edge_weight

        out = self.alpha * x
        h = x
        for _ in range(self.K):
            h = self.propagate(edge_index, x=h, edge_weight=edge_weight)
            out = out + ((1.0 - self.alpha) / self.K) * h

        return self.lin(out)

    def message(self, x_j, edge_weight=None):
        if edge_weight is None:
            return x_j
        return ops.expand_dims(edge_weight, -1) * x_j

