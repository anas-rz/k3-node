from typing import Dict, List, Optional, Tuple, Union
import math
import os
import os.path as osp

import keras
from keras import layers, ops

from k3_node.data.download import download_google_url


GROVER_PRETRAINED_MODELS = {
    "grover_base": {
        "google_id": "1hiGwOzoRfbJQPWj0V_mtOffsqIIAMgjl",
        "filename": "grover_base.pt",
        "hidden_size": 800,
        "num_attn_head": 4,
        "depth": 6,
        "num_mt_block": 1,
        "node_fdim": 151,
        "edge_fdim": 165,
    },
    "grover_large": {
        "google_id": "1bMg_ntUKEoOmHM0KoUi1XYJvzPBnHeWw",
        "filename": "grover_large.pt",
        "hidden_size": 1200,
        "num_attn_head": 6,
        "depth": 6,
        "num_mt_block": 1,
        "node_fdim": 151,
        "edge_fdim": 165,
    },
}


class GroverPReLU(layers.Layer):
    """PReLU activation layer with a learnable scalar alpha parameter matching PyTorch nn.PReLU(1)."""

    def __init__(self, init_val: float = 0.25, **kwargs):
        super().__init__(**kwargs)
        self.init_val = init_val
        self.weight = self.add_weight(
            name="weight",
            shape=(1,),
            initializer=keras.initializers.Constant(self.init_val),
            trainable=True,
        )

    def call(self, x):
        return ops.where(x >= 0, x, self.weight * x)

    def get_config(self):
        config = super().get_config()
        config.update({"init_val": self.init_val})
        return config


def get_activation(activation: str):
    """Get activation layer by name."""
    act_lower = activation.lower()
    if act_lower == "prelu":
        return GroverPReLU()
    elif act_lower == "relu":
        return layers.Activation("relu")
    elif act_lower == "leakyrelu":
        return layers.LeakyReLU(negative_slope=0.1)
    elif act_lower == "tanh":
        return layers.Activation("tanh")
    elif act_lower == "selu":
        return layers.Activation("selu")
    elif act_lower == "elu":
        return layers.Activation("elu")
    elif act_lower == "linear":
        return layers.Activation("linear")
    else:
        return layers.Activation(activation)


class MPNEncoder(layers.Layer):
    """Message Passing Neural Network encoder for atom or directed bond messages."""

    def __init__(
        self,
        hidden_size: int,
        depth: int = 6,
        atom_messages: bool = False,
        dropout: float = 0.0,
        undirected: bool = False,
        dense: bool = False,
        activation: str = "PReLU",
        input_layer: str = "none",
        input_dim: Optional[int] = None,
        bias: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.depth = depth
        self.atom_messages = atom_messages
        self.dropout_rate = dropout
        self.undirected = undirected
        self.dense = dense
        self.activation_name = activation
        self.input_layer_type = input_layer
        self.input_dim = input_dim
        self.bias = bias

    def build(self, input_shape=None):
        if self.input_layer_type == "fc":
            self.W_i = layers.Dense(self.hidden_size, use_bias=self.bias, name="W_i")
            if self.input_dim is not None:
                self.W_i.build((None, self.input_dim))
        self.W_h = layers.Dense(self.hidden_size, use_bias=self.bias, name="W_h")
        self.W_h.build((None, self.hidden_size))
        self.act_func = get_activation(self.activation_name)
        self.dropout_layer = layers.Dropout(self.dropout_rate)
        super().build(input_shape)

    def call(
        self,
        init_messages,
        init_attached_features,
        a2nei,
        a2attached,
        b2a=None,
        b2revb=None,
        training: bool = False,
    ):
        if self.input_layer_type == "fc":
            msg = self.act_func(self.W_i(init_messages))
        else:
            msg = init_messages

        input_msg = msg

        for _ in range(self.depth - 1):
            if self.undirected and b2revb is not None:
                rev = ops.take(msg, b2revb, axis=0)
                msg = (msg + rev) / 2.0

            nei_msg = ops.take(msg, a2nei, axis=0)
            nei_sum = ops.sum(nei_msg, axis=1)

            if not self.atom_messages:
                # Directed bond message passing (non-backtracking)
                a_msg = ops.take(nei_sum, b2a, axis=0)
                rev_msg = ops.take(msg, b2revb, axis=0)
                msg = a_msg - rev_msg
            else:
                msg = nei_sum

            msg = self.W_h(msg)

            if self.dense:
                msg = self.act_func(msg)
            else:
                msg = self.act_func(input_msg + msg)

            msg = self.dropout_layer(msg, training=training)

        return msg


class Head(layers.Layer):
    """Head containing query, key, and value MPN encoders."""

    def __init__(
        self,
        hidden_size: int,
        depth: int = 6,
        atom_messages: bool = False,
        dropout: float = 0.0,
        undirected: bool = False,
        dense: bool = False,
        activation: str = "PReLU",
        bias: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.depth = depth
        self.atom_messages = atom_messages
        self.dropout_rate = dropout
        self.undirected = undirected
        self.dense = dense
        self.activation_name = activation
        self.bias = bias

    def build(self, input_shape=None):
        self.mpn_q = MPNEncoder(
            hidden_size=self.hidden_size,
            depth=self.depth,
            atom_messages=self.atom_messages,
            dropout=self.dropout_rate,
            undirected=self.undirected,
            dense=self.dense,
            activation=self.activation_name,
            input_layer="none",
            bias=self.bias,
            name="mpn_q",
        )
        self.mpn_q.build(None)
        self.mpn_k = MPNEncoder(
            hidden_size=self.hidden_size,
            depth=self.depth,
            atom_messages=self.atom_messages,
            dropout=self.dropout_rate,
            undirected=self.undirected,
            dense=self.dense,
            activation=self.activation_name,
            input_layer="none",
            bias=self.bias,
            name="mpn_k",
        )
        self.mpn_k.build(None)
        self.mpn_v = MPNEncoder(
            hidden_size=self.hidden_size,
            depth=self.depth,
            atom_messages=self.atom_messages,
            dropout=self.dropout_rate,
            undirected=self.undirected,
            dense=self.dense,
            activation=self.activation_name,
            input_layer="none",
            bias=self.bias,
            name="mpn_v",
        )
        self.mpn_v.build(None)
        super().build(input_shape)

    def call(self, f_atoms, f_bonds, a2b, a2a, b2a, b2revb, training: bool = False):
        if self.atom_messages:
            init_messages = f_atoms
            init_attached = f_bonds
            a2nei = a2a
            a2att = a2b
        else:
            init_messages = f_bonds
            init_attached = f_atoms
            a2nei = a2b
            a2att = a2a

        q = self.mpn_q(
            init_messages, init_attached, a2nei, a2att, b2a=b2a, b2revb=b2revb, training=training
        )
        k = self.mpn_k(
            init_messages, init_attached, a2nei, a2att, b2a=b2a, b2revb=b2revb, training=training
        )
        v = self.mpn_v(
            init_messages, init_attached, a2nei, a2att, b2a=b2a, b2revb=b2revb, training=training
        )
        return q, k, v


class MultiHeadedAttention(layers.Layer):
    """Multi-headed attention across MPN heads."""

    def __init__(
        self,
        num_heads: int,
        hidden_size: int,
        dropout: float = 0.1,
        bias: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.d_k = hidden_size // num_heads
        self.dropout_rate = dropout
        self.bias = bias

    def build(self, input_shape=None):
        # Q, K, V projections use bias=True in reference implementation
        self.linear_q = layers.Dense(self.hidden_size, use_bias=True, name="linear_layers_0")
        self.linear_q.build((None, self.hidden_size))
        self.linear_k = layers.Dense(self.hidden_size, use_bias=True, name="linear_layers_1")
        self.linear_k.build((None, self.hidden_size))
        self.linear_v = layers.Dense(self.hidden_size, use_bias=True, name="linear_layers_2")
        self.linear_v.build((None, self.hidden_size))
        self.output_linear = layers.Dense(self.hidden_size, use_bias=self.bias, name="output_linear")
        self.output_linear.build((None, self.hidden_size))
        self.dropout_layer = layers.Dropout(self.dropout_rate)
        super().build(input_shape)

    def call(self, query, key, value, mask=None, training: bool = False):
        batch_size = ops.shape(query)[0]
        seq_len = ops.shape(query)[1]

        q = self.linear_q(query)
        k = self.linear_k(key)
        v = self.linear_v(value)

        q = ops.reshape(q, (batch_size, seq_len, self.num_heads, self.d_k))
        q = ops.transpose(q, (0, 2, 1, 3))
        k = ops.reshape(k, (batch_size, seq_len, self.num_heads, self.d_k))
        k = ops.transpose(k, (0, 2, 1, 3))
        v = ops.reshape(v, (batch_size, seq_len, self.num_heads, self.d_k))
        v = ops.transpose(v, (0, 2, 1, 3))

        scores = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) / math.sqrt(float(self.d_k))
        if mask is not None:
            scores = ops.where(mask == 0, -1e9, scores)

        p_attn = ops.softmax(scores, axis=-1)
        p_attn = self.dropout_layer(p_attn, training=training)

        x = ops.matmul(p_attn, v)
        x = ops.transpose(x, (0, 2, 1, 3))
        x = ops.reshape(x, (batch_size, seq_len, self.hidden_size))
        return self.output_linear(x)


class MTBlock(layers.Layer):
    """Multi-headed Message Passing Transformer Block."""

    def __init__(
        self,
        hidden_size: int,
        input_dim: int,
        num_attn_head: int = 4,
        depth: int = 6,
        activation: str = "PReLU",
        dropout: float = 0.0,
        bias: bool = False,
        atom_messages: bool = False,
        res_connection: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.input_dim = input_dim
        self.num_attn_head = num_attn_head
        self.depth = depth
        self.activation_name = activation
        self.dropout_rate = dropout
        self.bias = bias
        self.atom_messages = atom_messages
        self.res_connection = res_connection

    def build(self, input_shape=None):
        self.act_func = get_activation(self.activation_name)
        self.dropout_layer = layers.Dropout(self.dropout_rate)
        self.layernorm = layers.LayerNormalization(axis=-1, epsilon=1e-5, name="layernorm")
        self.layernorm.build((None, self.hidden_size))

        self.W_i = layers.Dense(self.hidden_size, use_bias=self.bias, name="W_i")
        self.W_i.build((None, self.input_dim))

        self.attn = MultiHeadedAttention(
            num_heads=self.num_attn_head,
            hidden_size=self.hidden_size,
            dropout=self.dropout_rate,
            bias=self.bias,
            name="attn",
        )
        self.attn.build(None)

        self.W_o = layers.Dense(self.hidden_size, use_bias=self.bias, name="W_o")
        self.W_o.build((None, self.hidden_size * self.num_attn_head))

        self.sublayer_norm = layers.LayerNormalization(axis=-1, epsilon=1e-5, name="sublayer_norm")
        self.sublayer_norm.build((None, self.hidden_size))

        self.heads = [
            Head(
                hidden_size=self.hidden_size,
                depth=self.depth,
                atom_messages=self.atom_messages,
                dropout=self.dropout_rate,
                activation=self.activation_name,
                bias=self.bias,
                name=f"heads_{i}",
            )
            for i in range(self.num_attn_head)
        ]
        for h in self.heads:
            h.build(None)
        super().build(input_shape)

    def call(self, f_atoms, f_bonds, a2b, b2a, b2revb, a2a, training: bool = False):
        if self.atom_messages:
            if ops.shape(f_atoms)[1] != self.hidden_size:
                f_atoms = self.W_i(f_atoms)
                f_atoms = self.dropout_layer(self.layernorm(self.act_func(f_atoms)), training=training)
        else:
            if ops.shape(f_bonds)[1] != self.hidden_size:
                f_bonds = self.W_i(f_bonds)
                f_bonds = self.dropout_layer(self.layernorm(self.act_func(f_bonds)), training=training)

        queries, keys, values = [], [], []
        for head in self.heads:
            q, k, v = head(f_atoms, f_bonds, a2b, a2a, b2a, b2revb, training=training)
            queries.append(ops.expand_dims(q, axis=1))
            keys.append(ops.expand_dims(k, axis=1))
            values.append(ops.expand_dims(v, axis=1))

        queries = ops.concatenate(queries, axis=1)
        keys = ops.concatenate(keys, axis=1)
        values = ops.concatenate(values, axis=1)

        x_out = self.attn(queries, keys, values, training=training)
        n_items = ops.shape(x_out)[0]
        x_out = ops.reshape(x_out, (n_items, -1))
        x_out = self.W_o(x_out)

        x_in = None
        if self.res_connection:
            x_in = f_atoms if self.atom_messages else f_bonds

        norm_out = self.dropout_layer(self.sublayer_norm(x_out), training=training)
        res = norm_out if x_in is None else (x_in + norm_out)

        if self.atom_messages:
            f_atoms = res
        else:
            f_bonds = res

        return f_atoms, f_bonds


class PositionwiseFeedForward(layers.Layer):
    """Position-wise Feed-Forward Network."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        d_out: Optional[int] = None,
        activation: str = "PReLU",
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.d_ff = d_ff
        self.d_out = d_model if d_out is None else d_out
        self.activation_name = activation
        self.dropout_rate = dropout

    def build(self, input_shape=None):
        self.W_1 = layers.Dense(self.d_ff, use_bias=True, name="W_1")
        self.W_1.build((None, self.d_model))
        self.W_2 = layers.Dense(self.d_out, use_bias=True, name="W_2")
        self.W_2.build((None, self.d_ff))
        self.dropout_layer = layers.Dropout(self.dropout_rate)
        self.act_func = get_activation(self.activation_name)
        super().build(input_shape)

    def call(self, x, training: bool = False):
        return self.W_2(self.dropout_layer(self.act_func(self.W_1(x)), training=training))


class GTransEncoder(layers.Layer):
    """Dual-track Graph Transformer Encoder of GROVER."""

    def __init__(
        self,
        hidden_size: int = 800,
        edge_fdim: int = 165,
        node_fdim: int = 151,
        num_mt_block: int = 1,
        num_attn_head: int = 4,
        depth: int = 6,
        dropout: float = 0.0,
        activation: str = "PReLU",
        atom_emb_output: Optional[str] = "both",
        bias: bool = False,
        res_connection: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.edge_fdim = edge_fdim
        self.node_fdim = node_fdim
        self.num_mt_block = num_mt_block
        self.num_attn_head = num_attn_head
        self.depth = depth
        self.dropout_rate = dropout
        self.activation_name = activation
        self.atom_emb_output = atom_emb_output
        self.bias = bias
        self.res_connection = res_connection

    def build(self, input_shape=None):
        self.edge_blocks = []
        self.node_blocks = []

        edge_in_dim = self.edge_fdim
        node_in_dim = self.node_fdim

        for i in range(self.num_mt_block):
            e_dim = edge_in_dim if i == 0 else self.hidden_size
            n_dim = node_in_dim if i == 0 else self.hidden_size

            eb = MTBlock(
                hidden_size=self.hidden_size,
                input_dim=e_dim,
                num_attn_head=self.num_attn_head,
                depth=self.depth,
                activation=self.activation_name,
                dropout=self.dropout_rate,
                bias=self.bias,
                atom_messages=False,
                res_connection=self.res_connection,
                name=f"edge_blocks_{i}",
            )
            eb.build(None)
            self.edge_blocks.append(eb)

            nb = MTBlock(
                hidden_size=self.hidden_size,
                input_dim=n_dim,
                num_attn_head=self.num_attn_head,
                depth=self.depth,
                activation=self.activation_name,
                dropout=self.dropout_rate,
                bias=self.bias,
                atom_messages=True,
                res_connection=self.res_connection,
                name=f"node_blocks_{i}",
            )
            nb.build(None)
            self.node_blocks.append(nb)

        self.ffn_atom_from_atom = PositionwiseFeedForward(
            d_model=self.hidden_size + self.node_fdim,
            d_ff=self.hidden_size * 4,
            d_out=self.hidden_size,
            activation=self.activation_name,
            dropout=self.dropout_rate,
            name="ffn_atom_from_atom",
        )
        self.ffn_atom_from_atom.build(None)

        self.ffn_atom_from_bond = PositionwiseFeedForward(
            d_model=self.hidden_size + self.node_fdim,
            d_ff=self.hidden_size * 4,
            d_out=self.hidden_size,
            activation=self.activation_name,
            dropout=self.dropout_rate,
            name="ffn_atom_from_bond",
        )
        self.ffn_atom_from_bond.build(None)

        self.ffn_bond_from_atom = PositionwiseFeedForward(
            d_model=self.hidden_size + self.edge_fdim,
            d_ff=self.hidden_size * 4,
            d_out=self.hidden_size,
            activation=self.activation_name,
            dropout=self.dropout_rate,
            name="ffn_bond_from_atom",
        )
        self.ffn_bond_from_atom.build(None)

        self.ffn_bond_from_bond = PositionwiseFeedForward(
            d_model=self.hidden_size + self.edge_fdim,
            d_ff=self.hidden_size * 4,
            d_out=self.hidden_size,
            activation=self.activation_name,
            dropout=self.dropout_rate,
            name="ffn_bond_from_bond",
        )
        self.ffn_bond_from_bond.build(None)

        self.atom_from_atom_norm = layers.LayerNormalization(
            axis=-1, epsilon=1e-5, name="atom_from_atom_sublayer_norm"
        )
        self.atom_from_atom_norm.build((None, self.hidden_size))

        self.atom_from_bond_norm = layers.LayerNormalization(
            axis=-1, epsilon=1e-5, name="atom_from_bond_sublayer_norm"
        )
        self.atom_from_bond_norm.build((None, self.hidden_size))

        self.bond_from_atom_norm = layers.LayerNormalization(
            axis=-1, epsilon=1e-5, name="bond_from_atom_sublayer_norm"
        )
        self.bond_from_atom_norm.build((None, self.hidden_size))

        self.bond_from_bond_norm = layers.LayerNormalization(
            axis=-1, epsilon=1e-5, name="bond_from_bond_sublayer_norm"
        )
        self.bond_from_bond_norm.build((None, self.hidden_size))

        self.act_func_node = get_activation(self.activation_name)
        self.act_func_edge = get_activation(self.activation_name)
        self.dropout_layer = layers.Dropout(self.dropout_rate)
        super().build(input_shape)

    def _pointwise_to_atom(self, emb, atom_fea, index, ffn_layer):
        aggr = ops.take(emb, index, axis=0)
        aggr = ops.sum(aggr, axis=1)
        concat = ops.concatenate([atom_fea, aggr], axis=1)
        return ffn_layer(concat)

    def _pointwise_to_bond(self, emb, bond_fea, a2nei, b2revb_or_b2a_rev, ffn_layer):
        aggr = ops.take(emb, a2nei, axis=0)
        aggr = ops.sum(aggr, axis=1)
        rev = ops.take(emb, b2revb_or_b2a_rev, axis=0)
        aggr = aggr - rev
        concat = ops.concatenate([bond_fea, aggr], axis=1)
        return ffn_layer(concat)

    def call(self, f_atoms, f_bonds, a2b, b2a, b2revb, a2a, training: bool = False):
        orig_f_atoms = f_atoms
        orig_f_bonds = f_bonds

        # Node blocks (atom messages)
        node_atoms, node_bonds = f_atoms, f_bonds
        for nb in self.node_blocks:
            node_atoms, node_bonds = nb(
                node_atoms, node_bonds, a2b, b2a, b2revb, a2a, training=training
            )

        # Edge blocks (bond messages)
        edge_atoms, edge_bonds = f_atoms, f_bonds
        for eb in self.edge_blocks:
            edge_atoms, edge_bonds = eb(
                edge_atoms, edge_bonds, a2b, b2a, b2revb, a2a, training=training
            )

        atom_output = node_atoms
        bond_output = edge_bonds

        if self.atom_emb_output is None:
            return atom_output, bond_output

        # Atom embeddings
        atom_from_atom = self._pointwise_to_atom(
            atom_output, orig_f_atoms, a2a, self.ffn_atom_from_atom
        )
        atom_from_atom = self.dropout_layer(self.atom_from_atom_norm(atom_from_atom), training=training)

        atom_from_bond = self._pointwise_to_atom(
            bond_output, orig_f_atoms, a2b, self.ffn_atom_from_bond
        )
        atom_from_bond = self.dropout_layer(self.atom_from_bond_norm(atom_from_bond), training=training)

        # Bond embeddings
        # atom list for bond: concat [b2a[:, None], a2a[b2a]]
        b2a_exp = ops.expand_dims(b2a, axis=1)
        a2a_b2a = ops.take(a2a, b2a, axis=0)
        atom_list_for_bond = ops.concatenate([b2a_exp, a2a_b2a], axis=1)
        b2a_rev = ops.take(b2a, b2revb, axis=0)

        bond_from_atom = self._pointwise_to_bond(
            atom_output, orig_f_bonds, atom_list_for_bond, b2a_rev, self.ffn_bond_from_atom
        )
        bond_from_atom = self.dropout_layer(self.bond_from_atom_norm(bond_from_atom), training=training)

        bond_list_for_bond = ops.take(a2b, b2a, axis=0)
        bond_from_bond = self._pointwise_to_bond(
            bond_output, orig_f_bonds, bond_list_for_bond, b2revb, self.ffn_bond_from_bond
        )
        bond_from_bond = self.dropout_layer(self.bond_from_bond_norm(bond_from_bond), training=training)

        if self.atom_emb_output == "atom":
            return {
                "atom_from_atom": atom_from_atom,
                "atom_from_bond": atom_from_bond,
                "bond_from_atom": None,
                "bond_from_bond": None,
            }
        elif self.atom_emb_output == "bond":
            return {
                "atom_from_atom": None,
                "atom_from_bond": None,
                "bond_from_atom": bond_from_atom,
                "bond_from_bond": bond_from_bond,
            }
        else:
            return {
                "atom_from_atom": atom_from_atom,
                "atom_from_bond": atom_from_bond,
                "bond_from_atom": bond_from_atom,
                "bond_from_bond": bond_from_bond,
            }


class Readout(layers.Layer):
    """Scope-based Readout layer for graph-level representations."""

    def __init__(
        self,
        rtype: str = "mean",
        hidden_size: int = 800,
        attn_hidden: Optional[int] = None,
        attn_out: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.rtype = rtype
        self.hidden_size = hidden_size
        self.attn_hidden = attn_hidden
        self.attn_out = attn_out

    def build(self, input_shape=None):
        if self.rtype == "self_attention":
            self.w1 = self.add_weight(
                name="w1",
                shape=(self.attn_hidden, self.hidden_size),
                initializer="glorot_normal",
                trainable=True,
            )
            self.w2 = self.add_weight(
                name="w2",
                shape=(self.attn_out, self.attn_hidden),
                initializer="glorot_normal",
                trainable=True,
            )
        super().build(input_shape)

    def call(self, embeddings, scope):
        """Readout aggregation over molecules given scope (list or tensor of [start, size])."""
        # Convert scope to list of (start, size)
        if isinstance(scope, (list, tuple)):
            scope_list = scope
        else:
            # scope is tensor of shape (N, 2)
            scope_arr = ops.convert_to_numpy(scope)
            scope_list = [(int(row[0]), int(row[1])) for row in scope_arr]

        mol_vecs = []
        zero_vec = ops.zeros((self.hidden_size,), dtype=embeddings.dtype)

        for a_start, a_size in scope_list:
            if a_size == 0:
                if self.rtype == "self_attention":
                    mol_vecs.append(ops.zeros((self.attn_out * self.hidden_size,), dtype=embeddings.dtype))
                else:
                    mol_vecs.append(zero_vec)
            else:
                cur = embeddings[a_start : a_start + a_size]
                if self.rtype == "self_attention":
                    x = ops.tanh(ops.matmul(self.w1, ops.transpose(cur, (1, 0))))
                    x = ops.matmul(self.w2, x)
                    attn = ops.softmax(x, axis=-1)
                    pooled = ops.matmul(attn, cur)
                    mol_vecs.append(ops.reshape(pooled, (-1,)))
                else:
                    cur_mean = ops.sum(cur, axis=0) / float(a_size)
                    mol_vecs.append(cur_mean)

        return ops.stack(mol_vecs, axis=0)


class GROVER(keras.Model):
    """Complete GROVER Model."""

    def __init__(
        self,
        hidden_size: int = 800,
        edge_fdim: int = 165,
        node_fdim: int = 151,
        num_mt_block: int = 1,
        num_attn_head: int = 4,
        depth: int = 6,
        dropout: float = 0.0,
        activation: str = "PReLU",
        atom_emb_output: Optional[str] = "both",
        readout_type: str = "mean",
        bias: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.edge_fdim = edge_fdim
        self.node_fdim = node_fdim
        self.num_mt_block = num_mt_block
        self.num_attn_head = num_attn_head
        self.depth = depth
        self.dropout_rate = dropout
        self.activation_name = activation
        self.atom_emb_output = atom_emb_output
        self.readout_type = readout_type
        self.bias = bias

        self.encoders = GTransEncoder(
            hidden_size=hidden_size,
            edge_fdim=edge_fdim,
            node_fdim=node_fdim,
            num_mt_block=num_mt_block,
            num_attn_head=num_attn_head,
            depth=depth,
            dropout=dropout,
            activation=activation,
            atom_emb_output=atom_emb_output,
            bias=bias,
            name="encoders",
        )
        self.readout = Readout(rtype=readout_type, hidden_size=hidden_size, name="readout")

    def build(self, input_shape=None):
        self.encoders.build(None)
        self.readout.build(None)
        super().build(input_shape)

    def call(self, inputs, training: bool = False):
        """Inputs can be a tuple/list: (f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a)."""
        if isinstance(inputs, (list, tuple)):
            f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a = inputs
        elif isinstance(inputs, dict):
            f_atoms = inputs["f_atoms"]
            f_bonds = inputs["f_bonds"]
            a2b = inputs["a2b"]
            b2a = inputs["b2a"]
            b2revb = inputs["b2revb"]
            a_scope = inputs["a_scope"]
            b_scope = inputs["b_scope"]
            a2a = inputs["a2a"]
        else:
            raise ValueError("inputs must be a tuple, list, or dict of molecular tensors.")

        emb_dict = self.encoders(
            f_atoms, f_bonds, a2b, b2a, b2revb, a2a, training=training
        )
        return emb_dict

    def get_fingerprint(
        self,
        inputs,
        fingerprint_source: str = "both",
        features_batch: Optional[any] = None,
        training: bool = False,
    ):
        """Generate molecule-level fingerprints using Readout."""
        if isinstance(inputs, (list, tuple)):
            _, _, _, _, _, a_scope, b_scope, _ = inputs
        else:
            a_scope = inputs["a_scope"]
            b_scope = inputs["b_scope"]

        emb = self(inputs, training=training)
        atom_from_atom = self.readout(emb["atom_from_atom"], a_scope)
        atom_from_bond = self.readout(emb["atom_from_bond"], a_scope)

        if fingerprint_source == "atom":
            fp = ops.concatenate([atom_from_atom, atom_from_bond], axis=1)
        elif fingerprint_source == "bond":
            bond_from_atom = self.readout(emb["bond_from_atom"], b_scope)
            bond_from_bond = self.readout(emb["bond_from_bond"], b_scope)
            fp = ops.concatenate([bond_from_atom, bond_from_bond], axis=1)
        else:
            bond_from_atom = self.readout(emb["bond_from_atom"], b_scope)
            bond_from_bond = self.readout(emb["bond_from_bond"], b_scope)
            fp = ops.concatenate(
                [atom_from_atom, atom_from_bond, bond_from_atom, bond_from_bond], axis=1
            )

        if features_batch is not None:
            fp = ops.concatenate([fp, features_batch], axis=1)

        return fp


def load_grover_weights(model: GROVER, checkpoint_path: str):
    """Loads PyTorch GROVER checkpoint state dict into Keras 3 GROVER model."""
    import torch

    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception:
        ckpt = torch.load(checkpoint_path, map_location="cpu")

    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    def get_np(key):
        t = state_dict[key]
        return t.detach().cpu().float().numpy()

    # Build model if not built
    if not model.built:
        model.build(None)

    enc = model.encoders

    # Load edge_blocks and node_blocks
    def load_mt_blocks(blocks, block_name):
        for bi, block in enumerate(blocks):
            prefix = f"grover.encoders.{block_name}.{bi}"

            # W_i
            w_i = get_np(f"{prefix}.W_i.weight")
            block.W_i.kernel.assign(w_i.T)

            # act_func
            block.act_func.weight.assign(get_np(f"{prefix}.act_func.weight"))

            # layernorm
            block.layernorm.gamma.assign(get_np(f"{prefix}.layernorm.weight"))
            block.layernorm.beta.assign(get_np(f"{prefix}.layernorm.bias"))

            # sublayer norm
            block.sublayer_norm.gamma.assign(get_np(f"{prefix}.sublayer.norm.weight"))
            block.sublayer_norm.beta.assign(get_np(f"{prefix}.sublayer.norm.bias"))

            # W_o
            w_o = get_np(f"{prefix}.W_o.weight")
            block.W_o.kernel.assign(w_o.T)

            # attn linear layers
            q_w = get_np(f"{prefix}.attn.linear_layers.0.weight")
            q_b = get_np(f"{prefix}.attn.linear_layers.0.bias")
            block.attn.linear_q.kernel.assign(q_w.T)
            block.attn.linear_q.bias.assign(q_b)

            k_w = get_np(f"{prefix}.attn.linear_layers.1.weight")
            k_b = get_np(f"{prefix}.attn.linear_layers.1.bias")
            block.attn.linear_k.kernel.assign(k_w.T)
            block.attn.linear_k.bias.assign(k_b)

            v_w = get_np(f"{prefix}.attn.linear_layers.2.weight")
            v_b = get_np(f"{prefix}.attn.linear_layers.2.bias")
            block.attn.linear_v.kernel.assign(v_w.T)
            block.attn.linear_v.bias.assign(v_b)

            # attn output linear
            out_w = get_np(f"{prefix}.attn.output_linear.weight")
            block.attn.output_linear.kernel.assign(out_w.T)

            # heads
            for hi, head in enumerate(block.heads):
                h_prefix = f"{prefix}.heads.{hi}"
                for mpn_name in ["mpn_q", "mpn_k", "mpn_v"]:
                    mpn = getattr(head, mpn_name)
                    w_h = get_np(f"{h_prefix}.{mpn_name}.W_h.weight")
                    mpn.W_h.kernel.assign(w_h.T)
                    act_w = get_np(f"{h_prefix}.{mpn_name}.act_func.weight")
                    mpn.act_func.weight.assign(act_w)

    load_mt_blocks(enc.edge_blocks, "edge_blocks")
    load_mt_blocks(enc.node_blocks, "node_blocks")

    # Load FFNs
    ffn_names = [
        ("ffn_atom_from_atom", enc.ffn_atom_from_atom),
        ("ffn_atom_from_bond", enc.ffn_atom_from_bond),
        ("ffn_bond_from_atom", enc.ffn_bond_from_atom),
        ("ffn_bond_from_bond", enc.ffn_bond_from_bond),
    ]
    for ffn_key, ffn_layer in ffn_names:
        prefix = f"grover.encoders.{ffn_key}"
        w1 = get_np(f"{prefix}.W_1.weight")
        b1 = get_np(f"{prefix}.W_1.bias")
        w2 = get_np(f"{prefix}.W_2.weight")
        b2 = get_np(f"{prefix}.W_2.bias")
        act = get_np(f"{prefix}.act_func.weight")

        ffn_layer.W_1.kernel.assign(w1.T)
        ffn_layer.W_1.bias.assign(b1)
        ffn_layer.W_2.kernel.assign(w2.T)
        ffn_layer.W_2.bias.assign(b2)
        ffn_layer.act_func.weight.assign(act)

    # Sublayer layer norms
    enc.atom_from_atom_norm.gamma.assign(get_np("grover.encoders.atom_from_atom_sublayer.norm.weight"))
    enc.atom_from_atom_norm.beta.assign(get_np("grover.encoders.atom_from_atom_sublayer.norm.bias"))

    enc.atom_from_bond_norm.gamma.assign(get_np("grover.encoders.atom_from_bond_sublayer.norm.weight"))
    enc.atom_from_bond_norm.beta.assign(get_np("grover.encoders.atom_from_bond_sublayer.norm.bias"))

    enc.bond_from_atom_norm.gamma.assign(get_np("grover.encoders.bond_from_atom_sublayer.norm.weight"))
    enc.bond_from_atom_norm.beta.assign(get_np("grover.encoders.bond_from_atom_sublayer.norm.bias"))

    enc.bond_from_bond_norm.gamma.assign(get_np("grover.encoders.bond_from_bond_sublayer.norm.weight"))
    enc.bond_from_bond_norm.beta.assign(get_np("grover.encoders.bond_from_bond_sublayer.norm.bias"))

    # act_func_node and act_func_edge
    enc.act_func_node.weight.assign(get_np("grover.encoders.act_func_node.weight"))
    enc.act_func_edge.weight.assign(get_np("grover.encoders.act_func_edge.weight"))


def download_grover_checkpoint(
    checkpoint_name: str = "grover_base",
    cache_dir: Optional[str] = None,
) -> str:
    """Download official GROVER pre-trained model checkpoint from Google Drive."""
    if checkpoint_name not in GROVER_PRETRAINED_MODELS:
        raise ValueError(
            f"Unknown checkpoint '{checkpoint_name}'. Supported: {list(GROVER_PRETRAINED_MODELS.keys())}"
        )

    info = GROVER_PRETRAINED_MODELS[checkpoint_name]
    if cache_dir is None:
        cache_dir = osp.expanduser("~/.cache/k3_node/grover")

    os.makedirs(cache_dir, exist_ok=True)
    target_path = osp.join(cache_dir, info["filename"])

    if osp.exists(target_path) and osp.getsize(target_path) > 1000:
        return target_path

    # Check if previously downloaded in /tmp/grover_download_test
    tmp_path = osp.join("/tmp/grover_download_test", info["filename"])
    if osp.exists(tmp_path) and osp.getsize(tmp_path) > 1000:
        import shutil

        shutil.copyfile(tmp_path, target_path)
        return target_path

    download_google_url(info["google_id"], cache_dir, info["filename"])
    return target_path

