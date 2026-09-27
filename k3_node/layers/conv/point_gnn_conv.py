from typing import Callable
from keras import ops

from k3_node.layers.conv.message_passing import MessagePassing


class PointGNNConv(MessagePassing):
    r"""The PointGNN graph convolutional operator from the
    `"Point-GNN: Graph Neural Network for 3D Object Detection in a Point Cloud"
    <https://arxiv.org/abs/2003.01251>`_ paper.

    Example:
        ```python
        import numpy as np
        import keras
        from k3_node.layers import PointGNNConv

        x = np.random.rand(10, 8).astype("float32")  # 10 nodes with 8 features each
        edge_index = np.random.randint(0, 10, size=(2, 30))  # 30 random edges
        pos = np.random.rand(10, 3).astype("float32")  # 3D node positions

        layer = PointGNNConv(
            mlp_h=keras.layers.Dense(3),  # predicts a position offset per node
            mlp_f=keras.layers.Dense(16),  # edge feature network
            mlp_g=keras.layers.Dense(8),  # node update network (output size = input features)
        )
        out = layer(x, pos, edge_index)
        print(tuple(out.shape))  # (10, 8)
        ```
    """
    def __init__(
        self,
        mlp_h: Callable,
        mlp_f: Callable,
        mlp_g: Callable,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "max")
        super().__init__(node_dim=0, **kwargs)

        self.mlp_h = mlp_h
        self.mlp_f = mlp_f
        self.mlp_g = mlp_g

    def build(self, input_shape=None):
        self.built = True

    def call(self, inputs, pos=None, edge_index=None, **kwargs):
        if edge_index is None:
            if isinstance(inputs, (list, tuple)):
                if len(inputs) == 3:
                    x, pos, edge_index = inputs
                elif len(inputs) == 2:
                    x, edge_index = inputs
                else:
                    raise ValueError(f"Unexpected input length {len(inputs)}")
            else:
                raise ValueError("Expected (x, pos, edge_index)")
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

        num_nodes = ops.shape(pos_dst)[0]
        out = self.propagate(
            edge_index,
            x=(x_src, x_dst),
            pos=(pos_src, pos_dst),
            size=(ops.shape(pos_src)[0], num_nodes),
        )

        out = self.mlp_g(out)
        return x_dst + out

    def message(self, pos_j, pos_i, x_i, x_j):
        delta = self.mlp_h(x_i)
        e = ops.concatenate([pos_j - pos_i + delta, x_j], axis=-1)
        return self.mlp_f(e)

