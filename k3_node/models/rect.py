from typing import Optional

import keras
from keras import ops

from k3_node.layers.conv import GCNConv
from k3_node.layers.conv.utils import scatter


class RECT_L(keras.layers.Layer):
    r"""The RECT model, *i.e.* its supervised RECT-L part, from the
    `"Network Embedding with Completely-imbalanced Labels"
    <https://arxiv.org/abs/2007.03545>`_ paper.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Intermediate size of each sample.
        normalize (bool, optional): Whether to add self-loops and compute
            symmetric normalization coefficients on-the-fly.
            (default: :obj:`True`)
        dropout (float, optional): The dropout probability.
            (default: :obj:`0.0`)
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        normalize: bool = True,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.dropout = dropout

        self.conv = GCNConv(in_channels, hidden_channels, normalize=normalize)
        self.lin = keras.layers.Dense(in_channels)
        self._dropout = keras.layers.Dropout(dropout) if dropout > 0 else None

    def build(self, input_shape=None):
        self.built = True

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.conv.reset_parameters()
        if self.lin.built:
            self.lin.kernel.assign(
                keras.initializers.GlorotUniform()(self.lin.kernel.shape)
            )
            if self.lin.bias is not None:
                self.lin.bias.assign(ops.zeros(self.lin.bias.shape))

    def call(self, x, edge_index, edge_weight=None, training=None):
        x = self.conv(x, edge_index, edge_weight=edge_weight)
        if self._dropout is not None:
            x = self._dropout(x, training=training)
        return self.lin(x)

    def embed(self, x, edge_index, edge_weight=None):
        return self.conv(x, edge_index, edge_weight=edge_weight)

    def get_semantic_labels(self, x, y, mask):
        r"""Replaces the original labels by their class-centers."""
        mask_shape = ops.shape(mask)
        if len(mask_shape) == 1 and 'bool' in str(mask.dtype):
            y_sub = y[mask]
            x_sub = x[mask]
        else:
            y_sub = ops.take(y, mask, axis=0)
            x_sub = ops.take(x, mask, axis=0)

        num_classes = int(ops.max(y_sub)) + 1
        mean = scatter(x_sub, y_sub, dim=0, dim_size=num_classes, reduce='mean')
        return ops.take(mean, y_sub, axis=0)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.hidden_channels})')
