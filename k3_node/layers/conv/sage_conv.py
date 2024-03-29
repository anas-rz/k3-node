# ported from https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/dense/dense_sage_conv.py
import keras
from keras import layers, ops


class SAGEConv(layers.Layer):
    """
    `k3_node.layers.SAGEConv`
    Implementation of GraphSAGE layer

    Args:
        out_channels: The number of output channels.
        normalize: Whether to normalize the output.
        bias: Whether to add a bias to the linear transformation.
    """
    def __init__(self, out_channels, normalize=False, bias=True):
        super().__init__()
        self.out_channels = out_channels
        self.normalize = normalize

        self.lin_rel = layers.Dense(out_channels, use_bias=False)
        self.lin_root = layers.Dense(out_channels, use_bias=bias)

    def call(self, x, adj, mask=None):
        # x = ops.expand_dims(x, axis=0) if len(ops.shape(x)) == 2 else x
        # adj = ops.expand_dims(adj, axis=0) if len(ops.shape(adj)) == 2 else adj

        out = ops.matmul(adj, x)
        out = out / ops.clip(
            ops.sum(adj, axis=-1, keepdims=True), x_min=1.0, x_max=float("inf")
        )
        out = self.lin_rel(out) + self.lin_root(x)

        if self.normalize:
            out = keras.utils.normalize(out, axis=-1)
        if mask is not None:
            mask = ops.expand_dims(mask, axis=-1)
            out = ops.multiply(out, mask)

        return out
