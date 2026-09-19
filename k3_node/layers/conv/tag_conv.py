from keras import layers, ops

from k3_node.layers.conv.message_passing import MessagePassing
from k3_node.layers.conv.utils import gcn_norm


class TAGConv(MessagePassing):
    r"""The topology adaptive graph convolutional operator from the
    `"Topology Adaptive Graph Convolutional Networks"
    <https://arxiv.org/abs/1710.10370>`_ paper.

    Args:
        in_channels: Size of each input sample.
        out_channels: Size of each output sample.
        K: Number of hops :math:`K`. (default: ``3``)
        bias: If set to :obj:`False`, the layer will not learn an additive bias.
            (default: ``True``)
        normalize: Whether to apply symmetric normalization. (default: ``True``)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int = 3,
        bias: bool = True,
        normalize: bool = True,
        **kwargs,
    ):
        super().__init__(aggr="add", **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.normalize = normalize
        self.use_bias = bias

        self.lins = [layers.Dense(out_channels, use_bias=False) for _ in range(K + 1)]

    def build(self, input_shape):
        feat_shape = input_shape[0] if isinstance(input_shape, (tuple, list)) and isinstance(input_shape[0], (tuple, list)) else input_shape
        for lin in self.lins:
            lin.build(feat_shape)

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
        if edge_index is None and isinstance(x, (tuple, list)):
            x, edge_index = x[0], x[1]

        if self.normalize:
            num_nodes = int(ops.shape(x)[self.node_dim])
            edge_index, edge_weight = gcn_norm(
                edge_index,
                edge_weight,
                num_nodes=num_nodes,
                add_self_loops=False,
                flow=self.flow,
                dtype=x.dtype,
            )

        out = self.lins[0](x)
        h = x
        for k in range(1, self.K + 1):
            h = self.propagate(edge_index, x=h, edge_weight=edge_weight)
            out = out + self.lins[k](h)

        if self.bias is not None:
            out = out + self.bias
        return out

    def message(self, x_j, edge_weight=None):
        if edge_weight is None:
            return x_j
        return ops.expand_dims(edge_weight, -1) * x_j

