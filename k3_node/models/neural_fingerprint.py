from typing import Optional

import keras
from keras import ops

from k3_node.layers.conv import MFConv
from k3_node.layers.pool import global_add_pool


class NeuralFingerprint(keras.layers.Layer):
    r"""The Neural Fingerprint model from the
    `"Convolutional Networks on Graphs for Learning Molecular Fingerprints"
    <https://arxiv.org/abs/1509.09292>`__ paper to generate fingerprints
    of molecules.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        out_channels (int): Size of each output fingerprint.
        num_layers (int): Number of layers.
        **kwargs (optional): Additional arguments of
            :class:`~k3_node.layers.conv.MFConv`.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers

        self.convs = []
        for i in range(self.num_layers):
            in_c = self.in_channels if i == 0 else self.hidden_channels
            self.convs.append(MFConv(in_c, hidden_channels, **kwargs))

        self.lins = []
        for _ in range(self.num_layers):
            self.lins.append(keras.layers.Dense(out_channels, use_bias=False))

    def build(self, input_shape=None):
        self.built = True

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for conv in self.convs:
            if hasattr(conv, "reset_parameters"):
                conv.reset_parameters()
        for lin in self.lins:
            if lin.built:
                lin.kernel.assign(keras.initializers.GlorotUniform()(lin.kernel.shape))

    def call(
        self,
        x,
        edge_index,
        batch: Optional[any] = None,
        batch_size: Optional[int] = None,
    ):
        outs = []
        for conv, lin in zip(self.convs, self.lins):
            x = ops.sigmoid(conv(x, edge_index))
            y = ops.softmax(lin(x), axis=-1)
            outs.append(global_add_pool(y, batch, size=batch_size))

        out = outs[0]
        for item in outs[1:]:
            out = out + item
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, num_layers={self.num_layers})')

