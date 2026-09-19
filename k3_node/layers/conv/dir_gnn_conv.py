import copy
from typing import Optional
import keras
from keras import ops
from keras.layers import Layer, Dense

from k3_node.layers.conv.message_passing import MessagePassing


class DirGNNConv(Layer):
    r"""A directed graph neural network operator from the
    `"Directed Graph Neural Networks" <https://arxiv.org/abs/2301.07663>`_ paper.
    """
    def __init__(
        self,
        conv: MessagePassing,
        alpha: float = 0.5,
        root_weight: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.alpha = alpha
        self.root_weight = root_weight
        self.conv = conv

        try:
            self.conv_in = conv.__class__.from_config(conv.get_config())
            self.conv_out = conv.__class__.from_config(conv.get_config())
        except Exception:
            self.conv_in = copy.deepcopy(conv)
            self.conv_out = copy.deepcopy(conv)

        if hasattr(self.conv_in, "add_self_loops"):
            self.conv_in.add_self_loops = False
            self.conv_out.add_self_loops = False
        if hasattr(self.conv_in, "root_weight"):
            self.conv_in.root_weight = False
            self.conv_out.root_weight = False

        in_channels = getattr(conv, "in_channels", None)
        out_channels = getattr(conv, "out_channels", None)
        if isinstance(in_channels, (list, tuple)):
            in_channels = in_channels[0]
        self.in_channels = in_channels
        self.out_channels = out_channels

        if root_weight and out_channels is not None:
            self.lin = Dense(out_channels, use_bias=True)
        else:
            self.lin = None

    def build(self, input_shape=None):
        if self.in_channels is not None:
            if hasattr(self.conv_in, "build"):
                self.conv_in.build((None, self.in_channels))
            if hasattr(self.conv_out, "build"):
                self.conv_out.build((None, self.in_channels))
            if self.lin is not None:
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

        x_in = self.conv_in(x, edge_index)
        edge_index_rev = ops.stack([edge_index[1], edge_index[0]], axis=0)
        x_out = self.conv_out(x, edge_index_rev)

        out = self.alpha * x_out + (1.0 - self.alpha) * x_in

        if self.lin is not None:
            out = out + self.lin(x)

        return out
