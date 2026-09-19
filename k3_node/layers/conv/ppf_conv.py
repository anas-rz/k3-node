from typing import Optional, Callable
from keras import ops

from k3_node.layers.conv.message_passing import MessagePassing
from k3_node.layers.conv.utils import remove_self_loops, add_self_loops


def _cross_product_3d(v1, v2):
    return ops.stack(
        [
            v1[:, 1] * v2[:, 2] - v1[:, 2] * v2[:, 1],
            v1[:, 2] * v2[:, 0] - v1[:, 0] * v2[:, 2],
            v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0],
        ],
        axis=-1,
    )


def _get_angle(v1, v2):
    cross = _cross_product_3d(v1, v2)
    norm = ops.norm(cross, axis=-1)
    dot = ops.sum(v1 * v2, axis=-1)
    return ops.arctan2(norm, dot)


def point_pair_features(pos_i, pos_j, normal_i, normal_j):
    pseudo = pos_j - pos_i
    return ops.stack(
        [
            ops.norm(pseudo, axis=-1),
            _get_angle(normal_i, pseudo),
            _get_angle(normal_j, pseudo),
            _get_angle(normal_i, normal_j),
        ],
        axis=-1,
    )


class PPFConv(MessagePassing):
    r"""The PPFNet graph convolutional operator from the
    `"PPFNet: Global Context Aware Local Features for Robust 3D Point
    Matching" <https://arxiv.org/abs/1802.02669>`_ paper.
    """
    def __init__(
        self,
        local_nn: Optional[Callable] = None,
        global_nn: Optional[Callable] = None,
        add_self_loops: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "max")
        super().__init__(node_dim=0, **kwargs)

        self.local_nn = local_nn
        self.global_nn = global_nn
        self.add_self_loops = add_self_loops

    def build(self, input_shape=None):
        self.built = True

    def call(self, inputs, pos=None, normal=None, edge_index=None, **kwargs):
        if edge_index is None:
            if isinstance(inputs, (list, tuple)):
                if len(inputs) == 4:
                    x, pos, normal, edge_index = inputs
                elif len(inputs) == 3:
                    x, pos, edge_index = inputs
                else:
                    raise ValueError(f"Unexpected input length {len(inputs)}")
            else:
                raise ValueError("Expected inputs with edge_index")
        else:
            x = inputs

        if not self.built:
            self.build()

        if isinstance(x, (list, tuple)):
            x_src, x_dst = x
        else:
            x_src = x_dst = x

        if isinstance(pos, (list, tuple)):
            pos_src, pos_dst = pos
        else:
            pos_src = pos_dst = pos

        if isinstance(normal, (list, tuple)):
            norm_src, norm_dst = normal
        else:
            norm_src = norm_dst = normal

        num_nodes = ops.shape(pos_dst)[0]
        if self.add_self_loops:
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

        out = self.propagate(
            edge_index,
            x=(x_src, x_dst),
            pos=(pos_src, pos_dst),
            normal=(norm_src, norm_dst),
            size=(ops.shape(pos_src)[0], num_nodes),
        )

        if self.global_nn is not None:
            out = self.global_nn(out)

        return out

    def message(self, x_j=None, pos_i=None, pos_j=None, normal_i=None, normal_j=None):
        msg = point_pair_features(pos_i, pos_j, normal_i, normal_j)
        if x_j is not None:
            msg = ops.concatenate([x_j, msg], axis=-1)
        if self.local_nn is not None:
            msg = self.local_nn(msg)
        return msg

