from keras import layers, ops

from k3_node.layers.conv.message_passing import MessagePassing
from k3_node.layers.conv.utils import add_self_loops, remove_self_loops, degree


class ClusterGCNConv(MessagePassing):
    r"""The ClusterGCN graph convolutional operator from the
    `"Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph
    Convolutional Networks" <https://arxiv.org/abs/1905.07953>`_ paper.

    Args:
        in_channels: Size of each input sample.
        out_channels: Size of each output sample.
        diag_lambda: Diagonal enhancement coefficient :math:`\lambda`.
            (default: ``0.0``)
        add_self_loops: If set to :obj:`False`, will not add self-loops.
            (default: ``True``)
        bias: If set to :obj:`False`, the layer will not learn an additive bias.
            (default: ``True``)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        diag_lambda: float = 0.0,
        add_self_loops: bool = True,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(aggr="add", **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.diag_lambda = diag_lambda
        self.add_self_loops = add_self_loops
        self.use_bias = bias

        self.lin_out = layers.Dense(out_channels, use_bias=bias)
        self.lin_root = layers.Dense(out_channels, use_bias=False)

    def build(self, input_shape):
        feat_shape = input_shape[0] if isinstance(input_shape, (tuple, list)) and isinstance(input_shape[0], (tuple, list)) else input_shape
        self.lin_out.build(feat_shape)
        self.lin_root.build(feat_shape)
        self.built = True

    def call(self, x, edge_index=None, edge_weight=None, **kwargs):
        if edge_index is None and isinstance(x, (tuple, list)):
            x, edge_index = x[0], x[1]

        num_nodes = ops.shape(x)[self.node_dim]

        if self.add_self_loops:
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

        row, col = edge_index[0], edge_index[1]
        col_cast = ops.cast(col, "int32")
        deg = degree(col_cast, num_nodes=num_nodes)
        deg_inv = 1.0 / ops.maximum(ops.cast(deg, x.dtype), 1.0)

        edge_weight = ops.take(deg_inv, col_cast, axis=0)
        loop_mask = ops.equal(row, col)
        edge_weight = ops.where(loop_mask, edge_weight + self.diag_lambda * ops.take(deg_inv, col_cast, axis=0), edge_weight)

        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        return self.lin_out(out) + self.lin_root(x)

    def message(self, x_j, edge_weight=None):
        if edge_weight is None:
            return x_j
        return ops.expand_dims(edge_weight, -1) * x_j

