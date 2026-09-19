from typing import Optional, Union, List
from keras import ops

from k3_node.layers.conv.message_passing import MessagePassing
from k3_node.layers.aggr.base import Aggregation


class SimpleConv(MessagePassing):
    r"""A simple, parameter-free message passing operator.

    Args:
        aggr: The aggregation scheme to use (``"sum"``, ``"mean"``,
            ``"min"``, ``"max"``, ``"mul"``). (default: ``"sum"``)
        combine_root: The way to combine root node features with the
            aggregated output (``"sum"``, ``"cat"``, ``"self_loop"``,
            or :obj:`None`). (default: :obj:`None`)
    """

    def __init__(
        self,
        aggr: Union[str, List[str], Aggregation, None] = "sum",
        combine_root: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(aggr=aggr, **kwargs)
        self.combine_root = combine_root
        if combine_root is not None and combine_root not in ["sum", "cat", "self_loop"]:
            raise ValueError(
                f"combine_root must be 'sum', 'cat', 'self_loop', or None, got {combine_root}"
            )

    def build(self, input_shape):
        self.built = True

    def call(self, x, edge_index=None, edge_weight=None, size=None, **kwargs):
        if edge_index is None and isinstance(x, (tuple, list)):
            x, edge_index = x[0], x[1]

        if not isinstance(x, (tuple, list)):
            x_src, x_dst = x, x
        else:
            x_src, x_dst = x[0], x[1]

        if self.combine_root == "self_loop":
            from k3_node.layers.conv.utils import add_self_loops
            num_nodes = ops.shape(x_src)[0]
            edge_index, edge_weight = add_self_loops(edge_index, edge_weight, num_nodes=num_nodes)

        out = self.propagate(edge_index, x=(x_src, x_dst), edge_weight=edge_weight, size=size)

        if self.combine_root is not None and x_dst is not None and self.combine_root != "self_loop":
            if self.combine_root == "sum":
                out = out + x_dst
            elif self.combine_root == "cat":
                out = ops.concatenate([out, x_dst], axis=-1)

        return out

    def message(self, x_j, edge_weight=None):
        if edge_weight is None:
            return x_j
        return ops.expand_dims(edge_weight, -1) * x_j
