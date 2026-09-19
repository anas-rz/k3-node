from typing import Optional, Callable, Tuple
from keras import ops

from k3_node.layers.conv.message_passing import MessagePassing
from k3_node.layers.conv.utils import remove_self_loops, add_self_loops


class PointNetConv(MessagePassing):
    r"""The PointNet set abstraction layer from the `"PointNet++: Deep
    Hierarchical Feature Learning on Point Sets in a Metric Space"
    <https://arxiv.org/abs/1706.02413>`_ paper.
    """
    def __init__(
        self,
        local_nn: Optional[Callable] = None,
        global_nn: Optional[Callable] = None,
        add_self_loops: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "max")
        super().__init__(**kwargs)

        self.local_nn = local_nn
        self.global_nn = global_nn
        self.add_self_loops = add_self_loops

    def build(self, input_shape=None):
        self.built = True

    def call(self, inputs, pos=None, edge_index=None, **kwargs):
        if edge_index is None:
            if isinstance(inputs, (list, tuple)):
                if len(inputs) == 3:
                    x, pos, edge_index = inputs
                elif len(inputs) == 2:
                    # x can be None or pos
                    x, edge_index = inputs
                else:
                    raise ValueError(f"Unexpected input length {len(inputs)}")
            else:
                raise ValueError("Expected (x, pos, edge_index) or (x, edge_index)")
        else:
            x = inputs

        if not self.built:
            self.build()

        if isinstance(pos, (list, tuple)):
            pos_src, pos_dst = pos
        else:
            pos_src = pos_dst = pos

        if isinstance(x, (list, tuple)):
            x_src, x_dst = x
        else:
            x_src = x_dst = x

        num_nodes = ops.shape(pos_dst)[0]
        if self.add_self_loops:
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

        out = self.propagate(
            edge_index,
            x=(x_src, x_dst),
            pos=(pos_src, pos_dst),
            size=(ops.shape(pos_src)[0], num_nodes),
        )

        if self.global_nn is not None:
            out = self.global_nn(out)

        return out

    def message(self, x_j=None, pos_i=None, pos_j=None):
        msg = pos_j - pos_i
        if x_j is not None:
            msg = ops.concatenate([x_j, msg], axis=-1)
        if self.local_nn is not None:
            msg = self.local_nn(msg)
        return msg


PointConv = PointNetConv

