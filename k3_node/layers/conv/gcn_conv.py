from keras import layers, ops

from k3_node.layers.conv.message_passing import MessagePassing
from k3_node.layers.conv.utils import gcn_norm


class GCNConv(MessagePassing):
    r"""The graph convolutional operator from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper.

    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta}

    Args:
        in_channels: Size of each input sample.
        out_channels: Size of each output sample.
        improved: If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}} = \mathbf{A} + 2 \mathbf{I}`.
            (default: :obj:`False`)
        cached: If set to :obj:`True`, the layer will cache the computation of
            :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}} \mathbf{\hat{D}}^{-1/2}`.
            (default: :obj:`False`)
        add_self_loops: If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        normalize: Whether to add self-loops and compute
            symmetric normalization coefficients on the fly.
            (default: :obj:`True`)
        bias: If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: bool = True,
        normalize: bool = True,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(aggr="add", **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.use_bias = bias

        self.lin = layers.Dense(out_channels, use_bias=False)
        self._cached_edge_index = None
        self._cached_norm = None

    def build(self, input_shape):
        if isinstance(input_shape, (tuple, list)) and len(input_shape) > 0 and isinstance(input_shape[0], (tuple, list)):
            feat_shape = input_shape[0]
        else:
            feat_shape = input_shape
        self.lin.build(feat_shape)
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.out_channels,),
                initializer="zeros",
                name="bias",
            )
        else:
            self.bias = None
        self.built = True

    def call(self, x, edge_index=None, edge_weight=None, **kwargs):
        # Handle legacy call conv((x, adj))
        if edge_index is None and isinstance(x, (tuple, list)):
            x, edge_index = x[0], x[1]

        # Handle dense adjacency [N, N]
        if len(ops.shape(edge_index)) == 2 and ops.shape(edge_index)[0] > 2 and ops.shape(edge_index)[0] == ops.shape(edge_index)[1]:
            where_adj = ops.where(edge_index != 0)
            where_adj = where_adj if not isinstance(where_adj, list) else where_adj
            edge_weight = ops.take(edge_index, where_adj[0] * ops.shape(edge_index)[1] + where_adj[1]) if edge_weight is None else edge_weight
            edge_index = ops.stack([where_adj[0], where_adj[1]], axis=0)

        if self.normalize:
            if self.cached and self._cached_edge_index is not None:
                edge_index = self._cached_edge_index
                edge_weight = self._cached_norm
            else:
                num_nodes = int(ops.shape(x)[self.node_dim])
                edge_index, edge_weight = gcn_norm(
                    edge_index,
                    edge_weight,
                    num_nodes=num_nodes,
                    improved=self.improved,
                    add_self_loops=self.add_self_loops,
                    flow=self.flow,
                    dtype=x.dtype,
                )
                if self.cached:
                    self._cached_edge_index = edge_index
                    self._cached_norm = edge_weight

        x = self.lin(x)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        if self.bias is not None:
            out = out + self.bias
        return out

    def message(self, x_j, edge_weight=None):
        if edge_weight is None:
            return x_j
        return ops.expand_dims(edge_weight, -1) * x_j
