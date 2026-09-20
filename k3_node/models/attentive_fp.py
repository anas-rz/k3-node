from typing import Optional
import keras
from keras import ops

from k3_node.layers.conv import GATConv
from k3_node.layers.conv.message_passing import MessagePassing
from k3_node.layers.conv.utils import softmax
from k3_node.layers.pool import global_add_pool
from k3_node.layers.pool.glob import _infer_size

try:
    from keras.src.backend.common.symbolic_scope import in_symbolic_scope
except ImportError:
    def in_symbolic_scope():
        return False


def _gru_step(cell, x, h):
    out, _ = cell(x, [h])
    return out


class GATEConv(MessagePassing):
    r"""The edge-conditioned attention layer used as the first message
    passing step of `AttentiveFP`."""
    def __init__(self, in_channels: int, out_channels: int, edge_dim: int,
                dropout: float = 0.0, **kwargs):
        super().__init__(aggr="add", node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.dropout_rate = dropout

        self.lin1 = keras.layers.Dense(out_channels, use_bias=False)
        self.lin2 = keras.layers.Dense(out_channels, use_bias=False)
        self.dropout = keras.layers.Dropout(dropout) if dropout > 0.0 else None

        self.lin1.build((None, in_channels + edge_dim))
        self.lin2.build((None, out_channels))

        self.att_l = self.add_weight(shape=(1, out_channels), initializer="glorot_uniform", name="att_l")
        self.att_r = self.add_weight(shape=(1, in_channels), initializer="glorot_uniform", name="att_r")
        self.bias = self.add_weight(shape=(out_channels,), initializer="zeros", name="bias")

    def build(self, input_shape=None):
        self.built = True

    def call(self, x, edge_index, edge_attr, training=None):
        row, col = ops.cast(edge_index[0], "int32"), ops.cast(edge_index[1], "int32")

        x_j = ops.take(x, row, axis=0)
        x_i = ops.take(x, col, axis=0)

        edge_attr = ops.cast(edge_attr, x.dtype)
        h = ops.leaky_relu(self.lin1(ops.concatenate([x_j, edge_attr], axis=-1)), negative_slope=0.01)
        alpha_j = ops.sum(h * self.att_l, axis=-1)
        alpha_i = ops.sum(x_i * self.att_r, axis=-1)
        alpha = ops.leaky_relu(alpha_j + alpha_i, negative_slope=0.01)

        num_nodes = ops.shape(x)[0]
        alpha = softmax(alpha, col, num_nodes=num_nodes, dim=0)
        if self.dropout is not None:
            alpha = self.dropout(alpha, training=training)

        message = self.lin2(x_j) * ops.expand_dims(alpha, -1)
        out = ops.segment_sum(message, col, num_segments=num_nodes)
        return out + self.bias


class AttentiveFP(keras.Model):
    r"""The Attentive FP model for molecular representation learning from the
    `"Pushing the Boundaries of Molecular Representation for Drug Discovery
    with the Graph Attention Mechanism"
    <https://pubs.acs.org/doi/10.1021/acs.jmedchem.9b00959>`_ paper, based on
    graph attention mechanisms.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Hidden node feature dimensionality.
        out_channels (int): Size of each output sample.
        edge_dim (int): Edge feature dimensionality.
        num_layers (int): Number of GNN layers.
        num_timesteps (int): Number of iterative refinement steps for global
            readout.
        dropout (float, optional): Dropout probability. (default: `0.0`)
        batch_size (int, optional): Fixed batch size (number of graphs) for JAX/XLA
            static shape compatibility. (default: `None`)
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        edge_dim: int,
        num_layers: int,
        num_timesteps: int,
        dropout: float = 0.0,
        batch_size: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.num_timesteps = num_timesteps
        self.dropout_rate = dropout
        self.batch_size = batch_size

        self.lin1 = keras.layers.Dense(hidden_channels)
        self.lin1.build((None, in_channels))

        self.gate_conv = GATEConv(hidden_channels, hidden_channels, edge_dim, dropout)
        self.gru = keras.layers.GRUCell(hidden_channels)
        self.gru.build((None, hidden_channels))

        self.atom_convs = []
        self.atom_grus = []
        for _ in range(num_layers - 1):
            conv = GATConv(hidden_channels, hidden_channels, dropout=dropout,
                           add_self_loops=False, negative_slope=0.01)
            conv.build((None, hidden_channels))
            self.atom_convs.append(conv)
            gru = keras.layers.GRUCell(hidden_channels)
            gru.build((None, hidden_channels))
            self.atom_grus.append(gru)

        self.mol_conv = GATConv(hidden_channels, hidden_channels, dropout=dropout,
                                add_self_loops=False, negative_slope=0.01)
        self.mol_conv.build([(None, hidden_channels), (None, hidden_channels)])
        self.mol_gru = keras.layers.GRUCell(hidden_channels)
        self.mol_gru.build((None, hidden_channels))

        self.lin2 = keras.layers.Dense(out_channels)
        self.lin2.build((None, hidden_channels))

        self.dropout = keras.layers.Dropout(dropout) if dropout > 0.0 else None
        self.built = True

    def call(self, x, edge_index=None, edge_attr=None, batch=None, batch_size=None, training=None):
        if isinstance(x, dict):
            edge_index = x.get("edge_index")
            edge_attr = x.get("edge_attr")
            batch = x.get("batch")
            batch_size = x.get("batch_size", batch_size)
            x = x.get("x")
        elif isinstance(x, (tuple, list)) and edge_index is None:
            if len(x) >= 4:
                x, edge_index, edge_attr, batch = x[0], x[1], x[2], x[3]
            elif len(x) == 3:
                x, edge_index, edge_attr = x[0], x[1], x[2]

        bs = batch_size if batch_size is not None else self.batch_size
        x = ops.cast(x, "float32")
        if edge_attr is not None:
            edge_attr = ops.cast(edge_attr, "float32")
        # Atom Embedding:
        x = ops.leaky_relu(self.lin1(x), negative_slope=0.01)

        h = ops.elu(self.gate_conv(x, edge_index, edge_attr, training=training))
        if self.dropout is not None:
            h = self.dropout(h, training=training)
        x = ops.relu(_gru_step(self.gru, h, x))

        for conv, gru in zip(self.atom_convs, self.atom_grus):
            h = conv(x, edge_index)
            h = ops.elu(h)
            if self.dropout is not None:
                h = self.dropout(h, training=training)
            x = ops.relu(_gru_step(gru, h, x))

        # Molecule Embedding:
        if batch is None:
            batch = ops.zeros((ops.shape(x)[0],), dtype="int32")
        else:
            batch = ops.cast(batch, "int32")
        num_nodes = ops.shape(batch)[0]
        row = ops.arange(num_nodes, dtype="int32")
        mol_edge_index = ops.stack([row, batch], axis=0)

        if bs is not None or in_symbolic_scope():
            size = bs
        else:
            size = _infer_size(batch)
            if size is None:
                size = bs

        out = ops.relu(global_add_pool(x, batch, size=size))
        for _ in range(self.num_timesteps):
            h = ops.elu(self.mol_conv((x, out), mol_edge_index))
            if self.dropout is not None:
                h = self.dropout(h, training=training)
            out = ops.relu(_gru_step(self.mol_gru, h, out))

        # Predictor:
        if self.dropout is not None:
            out = self.dropout(out, training=training)
        return self.lin2(out)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"in_channels={self.in_channels}, "
            f"hidden_channels={self.hidden_channels}, "
            f"out_channels={self.out_channels}, "
            f"edge_dim={self.edge_dim}, "
            f"num_layers={self.num_layers}, "
            f"num_timesteps={self.num_timesteps}"
            f")"
        )
