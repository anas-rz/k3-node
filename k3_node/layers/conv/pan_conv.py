from typing import Optional, Tuple
import keras
from keras import ops
from keras.layers import Dense

from k3_node.layers.conv.message_passing import MessagePassing


class PANConv(MessagePassing):
    r"""The path integral based convolution operator from the
    `"Path Integral Based Convolution and Pooling for Graph Neural Networks"
    <https://arxiv.org/abs/2004.14805>`_ paper.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_size: int,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_size = filter_size

        self.lin = Dense(out_channels, use_bias=True)
        self.weight = self.add_weight(
            shape=(filter_size + 1,),
            initializer="glorot_uniform",
            name="weight",
        )

    def build(self, input_shape=None):
        self.lin.build((None, self.in_channels))
        self.built = True

    def call(self, inputs, edge_index=None, **kwargs):
        if edge_index is None:
            if isinstance(inputs, (list, tuple)) and len(inputs) == 2:
                x, edge_index = inputs
            else:
                raise ValueError("Expected (x, edge_index) or x and edge_index")
        else:
            x = inputs

        if not self.built:
            self.build()

        num_nodes = ops.shape(x)[0]

        # Construct adjacency matrix
        if hasattr(edge_index, "shape") and len(edge_index.shape) == 2 and edge_index.shape[0] == 2:
            row, col = edge_index[0], edge_index[1]
            adj = ops.scatter(
                ops.stack([row, col], axis=-1),
                ops.ones((ops.shape(row)[0],), dtype=x.dtype),
                shape=(num_nodes, num_nodes),
            )
        else:
            adj = ops.cast(edge_index, x.dtype)

        # PAN entropy / path calculation
        # M = sum_{k=0}^filter_size weight[k] * A^k
        eye = ops.eye(num_nodes, dtype=x.dtype)
        M = self.weight[0] * eye
        curr_adj = eye
        for k in range(1, self.filter_size + 1):
            curr_adj = ops.matmul(curr_adj, adj)
            M = M + self.weight[k] * curr_adj

        deg = ops.sum(M, axis=1)
        deg_inv_sqrt = ops.power(ops.maximum(deg, 1e-12), -0.5)
        # Avoid inf / nan
        deg_inv_sqrt = ops.where(ops.isfinite(deg_inv_sqrt), deg_inv_sqrt, 0.0)

        M_norm = ops.expand_dims(deg_inv_sqrt, 0) * M * ops.expand_dims(deg_inv_sqrt, 1)

        out = ops.matmul(M_norm, x)
        out = self.lin(out)

        return out, M_norm
