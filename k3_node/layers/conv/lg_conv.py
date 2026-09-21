from keras import ops

from k3_node.layers.conv.message_passing import MessagePassing
from k3_node.layers.conv.utils import gcn_norm


class LGConv(MessagePassing):
    r"""The LightGCN operator from the `"LightGCN: Simplifying and Powering
    Graph Convolution Network for Recommendation"
    <https://arxiv.org/abs/2002.02126>`_ paper.

    Args:
        normalize: Whether to apply symmetric normalization. (default: ``True``)
    """

    def __init__(self, normalize: bool = True, **kwargs):
        super().__init__(aggr="add", **kwargs)
        self.normalize = normalize

    def build(self, input_shape):
        self.built = True

    def call(self, x, edge_index=None, edge_weight=None, **kwargs):
        if edge_index is None and isinstance(x, (tuple, list)):
            x, edge_index = x[0], x[1]

        if self.normalize:
            num_nodes = x.shape[self.node_dim] if hasattr(x, "shape") and x.shape[self.node_dim] is not None else ops.shape(x)[self.node_dim]
            edge_index, edge_weight = gcn_norm(
                edge_index,
                edge_weight,
                num_nodes=num_nodes,
                add_self_loops=False,
                flow=self.flow,
                dtype=x.dtype,
            )

        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j, edge_weight=None):
        if edge_weight is None:
            return x_j
        return ops.expand_dims(edge_weight, -1) * x_j
