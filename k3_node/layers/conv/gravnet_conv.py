from typing import Optional, Union, Tuple
import keras
from keras import ops
from keras.layers import Dense

from k3_node.layers.conv.message_passing import MessagePassing
from k3_node.layers.pool.knn import knn_graph


class GravNetConv(MessagePassing):
    r"""The GravNet operator from the `"Learning Representations of Irregular
    Particle-Detector Geometry with Distance-Weighted Graph Networks"
    <https://arxiv.org/abs/1902.07987>`_ paper.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        space_dimensions: int,
        propagate_dimensions: int,
        k: int,
        num_workers: Optional[int] = None,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "mean")
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.space_dimensions = space_dimensions
        self.propagate_dimensions = propagate_dimensions
        self.k = k

        self.lin_s = Dense(space_dimensions, use_bias=True)
        self.lin_h = Dense(propagate_dimensions, use_bias=True)
        self.lin_out1 = Dense(out_channels, use_bias=True)
        self.lin_out2 = Dense(out_channels, use_bias=True)

    def build(self, input_shape=None):
        self.lin_s.build((None, self.in_channels))
        self.lin_h.build((None, self.in_channels))
        self.lin_out1.build((None, self.in_channels))
        self.lin_out2.build((None, self.propagate_dimensions))
        self.built = True

    def call(self, inputs, edge_index=None, **kwargs):
        if isinstance(inputs, (list, tuple)):
            x = inputs[0]
        else:
            x = inputs

        if not self.built:
            self.build()

        s = self.lin_s(x)
        h = self.lin_h(x)

        if edge_index is None:
            edge_index = knn_graph(s, k=self.k)

        edge_index = ops.cast(edge_index, "int32")
        s_src = ops.take(s, edge_index[0], axis=0)
        s_dst = ops.take(s, edge_index[1], axis=0)
        dist_sq = ops.sum(ops.square(s_src - s_dst), axis=-1)
        edge_weight = ops.exp(-10.0 * dist_sq)

        num_nodes = ops.shape(x)[0]
        out = self.propagate(
            edge_index,
            x=h,
            edge_weight=edge_weight,
            size=(num_nodes, num_nodes),
        )

        return self.lin_out1(x) + self.lin_out2(out)

    def message(self, x_j, edge_weight):
        return ops.expand_dims(edge_weight, -1) * x_j

