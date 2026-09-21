from keras import layers, ops

from k3_node.layers.conv.message_passing import MessagePassing
from k3_node.layers.conv.utils import gcn_norm, is_tracing


class APPNP(MessagePassing):
    r"""The approximate personalized propagation of neural predictions (APPNP)
    operator from the `"Predict then Propagate: Combining Neural Networks with
    Personalized PageRank for Classification on Graphs"
    <https://arxiv.org/abs/1810.05997>`_ paper.

    Args:
        K: Number of iterations :math:`K`.
        alpha: Teleport probability :math:`\alpha`.
        dropout: Dropout probability of edges or features during propagation.
            (default: ``0.0``)
        cached: If set to :obj:`True`, the layer will cache the computation of
            normalization coefficients. (default: ``False``)
        add_self_loops: If set to :obj:`False`, will not add self-loops.
            (default: ``True``)
        normalize: Whether to apply symmetric normalization. (default: ``True``)
    """

    def __init__(
        self,
        K: int,
        alpha: float,
        dropout: float = 0.0,
        cached: bool = False,
        add_self_loops: bool = True,
        normalize: bool = True,
        **kwargs,
    ):
        super().__init__(aggr="add", **kwargs)
        self.K = K
        self.alpha = alpha
        self.dropout_rate = dropout
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self._cached_edge_index = None
        self._cached_norm = None
        self.dropout = layers.Dropout(dropout) if dropout > 0.0 else None

    def build(self, input_shape=None):
        if self.dropout is not None and hasattr(self.dropout, "build"):
            self.dropout.build(input_shape)
        self.built = True

    def call(self, x, edge_index=None, edge_weight=None, **kwargs):
        if edge_index is None and isinstance(x, (tuple, list)):
            x, edge_index = x[0], x[1]

        if self.normalize:
            if self.cached and self._cached_edge_index is not None:
                edge_index = self._cached_edge_index
                edge_weight = self._cached_norm
            else:
                num_nodes = x.shape[self.node_dim] if hasattr(x, "shape") and x.shape[self.node_dim] is not None else ops.shape(x)[self.node_dim]
                edge_index, edge_weight = gcn_norm(
                    edge_index,
                    edge_weight,
                    num_nodes=num_nodes,
                    add_self_loops=self.add_self_loops,
                    flow=self.flow,
                    dtype=x.dtype,
                )
                if self.cached and not is_tracing(edge_index):
                    self._cached_edge_index = edge_index
                    self._cached_norm = edge_weight

        h = x
        for _ in range(self.K):
            if self.dropout is not None:
                h = self.dropout(h)
            h = self.propagate(edge_index, x=h, edge_weight=edge_weight)
            h = (1.0 - self.alpha) * h + self.alpha * x

        return h

    def message(self, x_j, edge_weight=None):
        if edge_weight is None:
            return x_j
        return ops.expand_dims(edge_weight, -1) * x_j


# Backward-compatible alias
APPNPConv = APPNP

