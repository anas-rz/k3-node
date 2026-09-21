import math
from typing import Optional, Union, Tuple
from keras import ops

from k3_node.layers.conv.message_passing import MessagePassing
from k3_node.layers.conv.utils import gcn_norm, is_tracing


class GCN2Conv(MessagePassing):
    r"""The graph convolutional operator from the `"Simple and Deep Graph
    Convolutional Networks" <https://arxiv.org/abs/2007.02133>`_ paper.

    Args:
        channels: Size of each input and output sample.
        alpha: The strength of the initial residual connection :math:`\alpha`.
        theta: The hyperparameter for the identity mapping :math:`\theta`.
            (default: :obj:`None`)
        layer: The layer index :math:`l`. (default: :obj:`None`)
        shared_weights: If set to :obj:`True`, will use the same weights
            for :math:`\mathbf{X}` and :math:`\mathbf{X}_0`. (default: :obj:`True`)
        cached: If set to :obj:`True`, will cache the computation of normalization
            coefficients. (default: ``False``)
        add_self_loops: If set to :obj:`False`, will not add self-loops.
            (default: ``True``)
        normalize: Whether to apply symmetric normalization. (default: ``True``)
    """

    def __init__(
        self,
        channels: int,
        alpha: float,
        theta: Optional[float] = None,
        layer: Optional[int] = None,
        shared_weights: bool = True,
        cached: bool = False,
        add_self_loops: bool = True,
        normalize: bool = True,
        **kwargs,
    ):
        super().__init__(aggr="add", **kwargs)
        self.channels = channels
        self.alpha = alpha
        self.beta = 1.0
        if theta is not None and layer is not None:
            self.beta = math.log(theta / layer + 1.0)
        self.cached = cached
        self.normalize = normalize
        self.add_self_loops = add_self_loops
        self.shared_weights = shared_weights

        self._cached_edge_index = None
        self._cached_norm = None

    def build(self, input_shape):
        self.weight1 = self.add_weight(
            shape=(self.channels, self.channels),
            initializer="glorot_uniform",
            name="weight1",
        )
        if not self.shared_weights:
            self.weight2 = self.add_weight(
                shape=(self.channels, self.channels),
                initializer="glorot_uniform",
                name="weight2",
            )
        else:
            self.weight2 = None
        self.built = True

    def call(self, x, x_0, edge_index=None, edge_weight=None, **kwargs):
        if edge_index is None and isinstance(x, (tuple, list)):
            x, x_0, edge_index = x[0], x[1], x[2]

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

        h = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        h = (1.0 - self.alpha) * h
        h_0 = self.alpha * x_0

        if self.weight2 is None:
            combined = h + h_0
            out = (1.0 - self.beta) * combined + self.beta * ops.matmul(combined, self.weight1)
        else:
            term1 = (1.0 - self.beta) * h + self.beta * ops.matmul(h, self.weight1)
            term2 = (1.0 - self.beta) * h_0 + self.beta * ops.matmul(h_0, self.weight2)
            out = term1 + term2

        return out

    def message(self, x_j, edge_weight=None):
        if edge_weight is None:
            return x_j
        return ops.expand_dims(edge_weight, -1) * x_j

