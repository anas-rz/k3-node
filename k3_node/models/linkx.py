import math
from typing import Optional

import keras
from keras import ops

from k3_node.layers.conv.message_passing import MessagePassing
from k3_node.layers.conv.utils import scatter
from k3_node.layers.norm import BatchNorm
from k3_node.models.mlp import MLP


class SparseLinear(keras.layers.Layer):
    r"""A sparse linear transformation operator computing :math:`\mathbf{A}\mathbf{W} + \mathbf{b}`."""
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_bias = bias

        self.weight = self.add_weight(
            shape=(in_channels, out_channels),
            initializer=keras.initializers.GlorotUniform(),
            trainable=True,
            name="weight",
        )
        if bias:
            self.bias = self.add_weight(
                shape=(out_channels,),
                initializer="zeros",
                trainable=True,
                name="bias",
            )
        else:
            self.bias = None

    def build(self, input_shape=None):
        self.built = True

    def reset_parameters(self):
        self.weight.assign(
            keras.initializers.GlorotUniform()(shape=(self.in_channels, self.out_channels))
        )
        if self.use_bias and self.bias is not None:
            self.bias.assign(ops.zeros((self.out_channels,)))

    def call(self, edge_index, edge_weight=None):
        row, col = edge_index[0], edge_index[1]
        weight_j = ops.take(self.weight, row, axis=0)
        if edge_weight is not None:
            weight_j = ops.expand_dims(edge_weight, -1) * weight_j

        out = scatter(weight_j, col, dim=0, dim_size=self.in_channels, reduce="sum")
        if self.use_bias and self.bias is not None:
            out = out + self.bias
        return out


class LINKX(keras.layers.Layer):
    r"""The LINKX model from the `"Large Scale Learning on Non-Homophilous
    Graphs: New Benchmarks and Strong Simple Methods"
    <https://arxiv.org/abs/2110.14446>`_ paper.

    Args:
        num_nodes (int): The number of nodes in the graph.
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        out_channels (int): Size of each output sample.
        num_layers (int): Number of layers of :math:`\textrm{MLP}_{f}`.
        num_edge_layers (int, optional): Number of layers of
            :math:`\textrm{MLP}_{\mathbf{A}}`. (default: :obj:`1`)
        num_node_layers (int, optional): Number of layers of
            :math:`\textrm{MLP}_{\mathbf{X}}`. (default: :obj:`1`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.0`)
    """
    def __init__(
        self,
        num_nodes: int,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        num_edge_layers: int = 1,
        num_node_layers: int = 1,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_edge_layers = num_edge_layers

        self.edge_lin = SparseLinear(num_nodes, hidden_channels)

        if num_edge_layers > 1:
            self.edge_norm = BatchNorm(hidden_channels)
            channels = [hidden_channels] * num_edge_layers
            self.edge_mlp = MLP(channels, dropout=0.0, act_first=True)
        else:
            self.edge_norm = None
            self.edge_mlp = None

        channels = [in_channels] + [hidden_channels] * num_node_layers
        self.node_mlp = MLP(channels, dropout=0.0, act_first=True)

        self.cat_lin1 = keras.layers.Dense(hidden_channels)
        self.cat_lin2 = keras.layers.Dense(hidden_channels)

        channels = [hidden_channels] * num_layers + [out_channels]
        self.final_mlp = MLP(channels, dropout=dropout, act_first=True)

    def build(self, input_shape=None):
        self.edge_lin.build()
        self.built = True

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.edge_lin.reset_parameters()
        if self.edge_norm is not None and hasattr(self.edge_norm, "reset_parameters"):
            self.edge_norm.reset_parameters()
        if self.edge_mlp is not None and hasattr(self.edge_mlp, "reset_parameters"):
            self.edge_mlp.reset_parameters()
        self.node_mlp.reset_parameters()
        self.final_mlp.reset_parameters()

    def call(self, x, edge_index, edge_weight=None, training=None):
        out = self.edge_lin(edge_index, edge_weight)

        if self.edge_norm is not None and self.edge_mlp is not None:
            out = ops.relu(out)
            out = self.edge_norm(out, training=training)
            out = self.edge_mlp(out, training=training)

        out = out + self.cat_lin1(out)

        if x is not None:
            x = self.node_mlp(x, training=training)
            out = out + x
            out = out + self.cat_lin2(x)

        return self.final_mlp(ops.relu(out), training=training)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(num_nodes={self.num_nodes}, '
                f'in_channels={self.in_channels}, '
                f'out_channels={self.out_channels})')

