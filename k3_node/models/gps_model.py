import os
import math
import zipfile
import urllib.request
from typing import Optional, List, Union, Tuple, Dict, Any

import numpy as np
import keras
from keras import layers, ops

from k3_node.layers.pool import global_add_pool, global_max_pool, global_mean_pool


def _get_act(act_str: str):
    act_str = act_str.lower()
    if act_str in ["gelu", "quick_gelu"]:
        return ops.gelu
    elif act_str == "relu":
        return ops.relu
    elif act_str == "silu" or act_str == "swish":
        return ops.silu
    elif act_str == "tanh":
        return ops.tanh
    elif act_str == "sigmoid":
        return ops.sigmoid
    return ops.relu


# 2022 OGB molecule atom feature dimensions (used in GraphGPS pretraining on PCQM4Mv2)
DEFAULT_ATOM_FEATURE_DIMS = [119, 4, 12, 12, 10, 6, 6, 2, 2]
DEFAULT_BOND_FEATURE_DIMS = [5, 6, 2]


class AtomEncoder(layers.Layer):
    r"""OGB Molecule categorical atom feature encoder.

    Args:
        emb_dim (int): Output embedding dimension.
        feature_dims (List[int], optional): Categorical feature vocabulary sizes for each
            atom feature column. (default: ``[119, 4, 12, 12, 10, 6, 6, 2, 2]``)
        **kwargs: Additional layer arguments.
    """

    def __init__(
        self,
        emb_dim: int,
        feature_dims: Optional[List[int]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.emb_dim = emb_dim
        self.feature_dims = feature_dims if feature_dims is not None else DEFAULT_ATOM_FEATURE_DIMS

        self.atom_embedding_list = [
            layers.Embedding(
                input_dim=dim,
                output_dim=emb_dim,
                embeddings_initializer="glorot_uniform",
                name=f"atom_embedding_{i}",
            )
            for i, dim in enumerate(self.feature_dims)
        ]

    def build(self, input_shape=None):
        for emb in self.atom_embedding_list:
            if not emb.built:
                emb.build(None)
        super().build(input_shape)

    def call(self, x):
        x = ops.cast(x, "int32")
        out = 0
        for i, emb in enumerate(self.atom_embedding_list):
            feat = x[:, i]
            out = out + emb(feat)
        return out


class BondEncoder(layers.Layer):
    r"""OGB Molecule categorical bond feature encoder.

    Args:
        emb_dim (int): Output embedding dimension.
        feature_dims (List[int], optional): Categorical feature vocabulary sizes for each
            bond feature column. (default: ``[5, 6, 2]``)
        **kwargs: Additional layer arguments.
    """

    def __init__(
        self,
        emb_dim: int,
        feature_dims: Optional[List[int]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.emb_dim = emb_dim
        self.feature_dims = feature_dims if feature_dims is not None else DEFAULT_BOND_FEATURE_DIMS

        self.bond_embedding_list = [
            layers.Embedding(
                input_dim=dim,
                output_dim=emb_dim,
                embeddings_initializer="glorot_uniform",
                name=f"bond_embedding_{i}",
            )
            for i, dim in enumerate(self.feature_dims)
        ]

    def build(self, input_shape=None):
        for emb in self.bond_embedding_list:
            if not emb.built:
                emb.build(None)
        super().build(input_shape)

    def call(self, edge_attr):
        edge_attr = ops.cast(edge_attr, "int32")
        out = 0
        for i, emb in enumerate(self.bond_embedding_list):
            feat = edge_attr[:, i]
            out = out + emb(feat)
        return out


class RWSEEncoder(layers.Layer):
    r"""Random Walk Structural Encoding (RWSE) node encoder.

    Normalizes the precomputed $k$-step diagonal random walk landing probabilities
    using Batch Normalization and projects them into `pe_dim` dimension.

    Args:
        num_rw_steps (int): Number of random walk steps. (default: ``16``)
        pe_dim (int): Output structural encoding dimension. (default: ``20``)
        **kwargs: Additional layer arguments.
    """

    def __init__(
        self,
        num_rw_steps: int = 16,
        pe_dim: int = 20,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_rw_steps = num_rw_steps
        self.pe_dim = pe_dim

        self.raw_norm = layers.BatchNormalization(
            axis=-1,
            epsilon=1e-5,
            momentum=0.9,
            name="raw_norm",
        )
        self.pe_encoder = layers.Dense(pe_dim, use_bias=True, name="pe_encoder")

    def build(self, input_shape=None):
        if not self.raw_norm.built:
            self.raw_norm.build((None, self.num_rw_steps))
        if not self.pe_encoder.built:
            self.pe_encoder.build((None, self.num_rw_steps))
        super().build(input_shape)

    def call(self, pestat_RWSE, training=False):
        pe = self.raw_norm(pestat_RWSE, training=training)
        return self.pe_encoder(pe)


class CustomGatedGCN(layers.Layer):
    r"""Residual Gated Graph ConvNet layer with edge feature updates.

    Reference:
    "Residual Gated Graph ConvNets" (Bresson & Laurent, 2017).

    Args:
        in_dim (int): Input feature dimension.
        out_dim (int): Output feature dimension.
        dropout (float, optional): Dropout rate. (default: ``0.0``)
        residual (bool, optional): Whether to use residual connections. (default: ``True``)
        act (str, optional): Activation function. (default: ``"gelu"``)
        **kwargs: Additional layer arguments.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.0,
        residual: bool = True,
        act: str = "gelu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout_rate = dropout
        self.residual = residual
        self.act_name = act

        self.A = layers.Dense(out_dim, use_bias=True, name="A")
        self.B = layers.Dense(out_dim, use_bias=True, name="B")
        self.C = layers.Dense(out_dim, use_bias=True, name="C")
        self.D = layers.Dense(out_dim, use_bias=True, name="D")
        self.E = layers.Dense(out_dim, use_bias=True, name="E")

        self.bn_node_x = layers.BatchNormalization(
            axis=-1, epsilon=1e-5, momentum=0.9, name="bn_node_x"
        )
        self.bn_edge_e = layers.BatchNormalization(
            axis=-1, epsilon=1e-5, momentum=0.9, name="bn_edge_e"
        )
        self.dropout_node = layers.Dropout(dropout)
        self.dropout_edge = layers.Dropout(dropout)

    def build(self, input_shape=None):
        if not self.A.built:
            self.A.build((None, self.in_dim))
            self.B.build((None, self.in_dim))
            self.C.build((None, self.in_dim))
            self.D.build((None, self.in_dim))
            self.E.build((None, self.in_dim))
            self.bn_node_x.build((None, self.out_dim))
            self.bn_edge_e.build((None, self.out_dim))
        super().build(input_shape)

    def call(self, x, edge_index, edge_attr, training=False):
        x_in = x
        e_in = edge_attr

        Ax = self.A(x)
        Bx = self.B(x)
        Ce = self.C(edge_attr)
        Dx = self.D(x)
        Ex = self.E(x)

        src = ops.cast(edge_index[0], "int32")
        dst = ops.cast(edge_index[1], "int32")

        Dx_i = ops.take(Dx, dst, axis=0)
        Ex_j = ops.take(Ex, src, axis=0)
        Bx_j = ops.take(Bx, src, axis=0)

        e_ij = Dx_i + Ex_j + Ce
        sigma_ij = ops.sigmoid(e_ij)

        num_nodes = ops.shape(x)[0]
        sum_sigma_x = ops.segment_sum(sigma_ij * Bx_j, dst, num_segments=num_nodes)
        sum_sigma = ops.segment_sum(sigma_ij, dst, num_segments=num_nodes)
        aggr_out = sum_sigma_x / (sum_sigma + 1e-6)

        x_out = self.bn_node_x(Ax + aggr_out, training=training)
        e_out = self.bn_edge_e(e_ij, training=training)

        act_fn = _get_act(self.act_name)
        x_out = act_fn(x_out)
        e_out = act_fn(e_out)

        x_out = self.dropout_node(x_out, training=training)
        e_out = self.dropout_edge(e_out, training=training)

        if self.residual:
            x_out = x_in + x_out
            e_out = e_in + e_out

        return x_out, e_out


class TransformerSelfAttention(layers.Layer):
    r"""Multi-Head Self-Attention layer matching PyTorch `nn.MultiheadAttention`.

    Args:
        embed_dim (int): Total embedding dimension.
        num_heads (int): Number of attention heads.
        dropout (float, optional): Attention dropout rate. (default: ``0.0``)
        **kwargs: Additional layer arguments.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = 1.0 / math.sqrt(self.head_dim)
        self.dropout_rate = dropout

        self.q_proj = layers.Dense(embed_dim, use_bias=True, name="q_proj")
        self.k_proj = layers.Dense(embed_dim, use_bias=True, name="k_proj")
        self.v_proj = layers.Dense(embed_dim, use_bias=True, name="v_proj")
        self.out_proj = layers.Dense(embed_dim, use_bias=True, name="out_proj")
        self.dropout = layers.Dropout(dropout)

    def build(self, input_shape=None):
        if not self.q_proj.built:
            self.q_proj.build((None, self.embed_dim))
            self.k_proj.build((None, self.embed_dim))
            self.v_proj.build((None, self.embed_dim))
            self.out_proj.build((None, self.embed_dim))
        super().build(input_shape)

    def call(self, x, mask=None, training=False):
        shape = ops.shape(x)
        bsz, n_node = shape[0], shape[1]

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = ops.transpose(
            ops.reshape(q, (bsz, n_node, self.num_heads, self.head_dim)),
            (0, 2, 1, 3),
        ) * self.scaling
        k = ops.transpose(
            ops.reshape(k, (bsz, n_node, self.num_heads, self.head_dim)),
            (0, 2, 1, 3),
        )
        v = ops.transpose(
            ops.reshape(v, (bsz, n_node, self.num_heads, self.head_dim)),
            (0, 2, 1, 3),
        )

        scores = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2)))  # [B, H, N, N]

        if mask is not None:
            # mask: [B, N] boolean tensor, True for valid tokens, False for padding
            # PyTorch key_padding_mask: True where padding
            key_padding_mask = ~mask
            scores = ops.where(
                ops.expand_dims(ops.expand_dims(key_padding_mask, axis=1), axis=2),
                float("-inf"),
                scores,
            )

        attn_weights = ops.softmax(scores, axis=-1)
        # Replace NaNs from all-inf rows
        attn_weights = ops.where(ops.isnan(attn_weights), 0.0, attn_weights)
        attn_weights = self.dropout(attn_weights, training=training)

        attn = ops.matmul(attn_weights, v)  # [B, H, N, head_dim]
        attn = ops.transpose(attn, (0, 2, 1, 3))
        attn = ops.reshape(attn, (bsz, n_node, self.embed_dim))

        return self.out_proj(attn)


class GPSLayer(layers.Layer):
    r"""GraphGPS hybrid layer combining local MPNN (e.g. `CustomGatedGCN`) and global Multi-Head Attention.

    Reference:
    "Recipe for a General, Powerful, Scalable Graph Transformer" (NeurIPS 2022).

    Args:
        dim_h (int): Hidden embedding dimension.
        local_gnn_type (str, optional): Local MPNN type. (default: ``"CustomGatedGCN"``)
        global_model_type (str, optional): Global attention type. (default: ``"Transformer"``)
        num_heads (int, optional): Number of attention heads. (default: ``8``)
        act (str, optional): Activation function. (default: ``"gelu"``)
        dropout (float, optional): Dropout rate. (default: ``0.0``)
        attn_dropout (float, optional): Attention dropout rate. (default: ``0.0``)
        layer_norm (bool, optional): Whether to use LayerNorm. (default: ``False``)
        batch_norm (bool, optional): Whether to use BatchNorm. (default: ``True``)
        **kwargs: Additional layer arguments.
    """

    def __init__(
        self,
        dim_h: int,
        local_gnn_type: str = "CustomGatedGCN",
        global_model_type: str = "Transformer",
        num_heads: int = 8,
        act: str = "gelu",
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        layer_norm: bool = False,
        batch_norm: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim_h = dim_h
        self.local_gnn_type = local_gnn_type
        self.global_model_type = global_model_type
        self.num_heads = num_heads
        self.act_name = act
        self.dropout_rate = dropout
        self.attn_dropout_rate = attn_dropout
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm

        if local_gnn_type == "CustomGatedGCN":
            self.local_model = CustomGatedGCN(
                in_dim=dim_h,
                out_dim=dim_h,
                dropout=dropout,
                residual=True,
                act=act,
                name="local_model",
            )
        else:
            self.local_model = None

        if global_model_type in ["Transformer", "BiasedTransformer"]:
            self.self_attn = TransformerSelfAttention(
                embed_dim=dim_h,
                num_heads=num_heads,
                dropout=attn_dropout,
                name="self_attn",
            )
        else:
            self.self_attn = None

        if layer_norm:
            self.norm1_local = layers.LayerNormalization(epsilon=1e-5, name="norm1_local")
            self.norm1_attn = layers.LayerNormalization(epsilon=1e-5, name="norm1_attn")
            self.norm2 = layers.LayerNormalization(epsilon=1e-5, name="norm2")
        elif batch_norm:
            self.norm1_local = layers.BatchNormalization(
                axis=-1, epsilon=1e-5, momentum=0.9, name="norm1_local"
            )
            self.norm1_attn = layers.BatchNormalization(
                axis=-1, epsilon=1e-5, momentum=0.9, name="norm1_attn"
            )
            self.norm2 = layers.BatchNormalization(
                axis=-1, epsilon=1e-5, momentum=0.9, name="norm2"
            )
        else:
            self.norm1_local = None
            self.norm1_attn = None
            self.norm2 = None

        self.dropout_local = layers.Dropout(dropout)
        self.dropout_attn = layers.Dropout(dropout)

        self.ff_linear1 = layers.Dense(dim_h * 2, use_bias=True, name="ff_linear1")
        self.ff_linear2 = layers.Dense(dim_h, use_bias=True, name="ff_linear2")
        self.ff_dropout1 = layers.Dropout(dropout)
        self.ff_dropout2 = layers.Dropout(dropout)

    def build(self, input_shape=None):
        if self.local_model is not None and not self.local_model.built:
            self.local_model.build(None)
        if self.self_attn is not None and not self.self_attn.built:
            self.self_attn.build(None)
        if self.norm1_local is not None and not self.norm1_local.built:
            self.norm1_local.build((None, self.dim_h))
        if self.norm1_attn is not None and not self.norm1_attn.built:
            self.norm1_attn.build((None, self.dim_h))
        if not self.ff_linear1.built:
            self.ff_linear1.build((None, self.dim_h))
            self.ff_linear2.build((None, self.dim_h * 2))
        if self.norm2 is not None and not self.norm2.built:
            self.norm2.build((None, self.dim_h))
        super().build(input_shape)

    def call(self, x, edge_index, edge_attr, batch=None, training=False):
        h_in1 = x
        h_out_list = []

        # Local MPNN
        if self.local_model is not None:
            h_local, edge_attr = self.local_model(
                x, edge_index, edge_attr, training=training
            )
            # CustomGatedGCN handles residual internally
            if self.norm1_local is not None:
                h_local = self.norm1_local(h_local, training=training)
            h_out_list.append(h_local)

        # Global Attention
        if self.self_attn is not None:
            if batch is None:
                h_dense = ops.expand_dims(x, axis=0)
                mask = ops.ones((1, ops.shape(x)[0]), dtype="bool")
                h_attn = self.self_attn(h_dense, mask=mask, training=training)[0]
            else:
                batch_np = ops.convert_to_numpy(batch)
                B_int = int(batch_np.max()) + 1 if len(batch_np) > 0 else 1
                counts = np.bincount(batch_np, minlength=B_int)
                max_nodes = int(counts.max()) if len(counts) > 0 else 0

                offsets = np.zeros(len(batch_np), dtype=np.int32)
                curr = np.zeros(B_int, dtype=np.int32)
                for i, b in enumerate(batch_np):
                    offsets[i] = curr[b]
                    curr[b] += 1

                offsets_t = ops.convert_to_tensor(offsets, dtype="int32")
                batch_cast = ops.cast(batch, "int32")
                indices = ops.stack([batch_cast, offsets_t], axis=1)

                dense_x = ops.scatter_update(
                    ops.zeros((B_int, max_nodes, self.dim_h), dtype=x.dtype),
                    indices,
                    x,
                )

                mask_np = np.zeros((B_int, max_nodes), dtype=bool)
                for b, count in enumerate(counts):
                    mask_np[b, :count] = True
                mask = ops.convert_to_tensor(mask_np)

                h_attn_dense = self.self_attn(dense_x, mask=mask, training=training)

                flat_attn = ops.reshape(h_attn_dense, (B_int * max_nodes, self.dim_h))
                flat_indices = batch_cast * max_nodes + offsets_t
                h_attn = ops.take(flat_attn, flat_indices, axis=0)

            h_attn = self.dropout_attn(h_attn, training=training)
            h_attn = h_in1 + h_attn
            if self.norm1_attn is not None:
                h_attn = self.norm1_attn(h_attn, training=training)
            h_out_list.append(h_attn)

        # Sum local and global representations
        h = sum(h_out_list)

        # Feed Forward block
        act_fn = _get_act(self.act_name)
        ff_out = self.ff_dropout1(act_fn(self.ff_linear1(h)), training=training)
        ff_out = self.ff_dropout2(self.ff_linear2(ff_out), training=training)
        h = h + ff_out

        if self.norm2 is not None:
            h = self.norm2(h, training=training)

        return h, edge_attr


class SANGraphHead(layers.Layer):
    r"""Prediction head for graph-level tasks from the Spectral Attention Network (SAN).

    Args:
        dim_in (int): Input feature dimension.
        dim_out (int): Output feature dimension. (default: ``1``)
        L (int, optional): Number of hidden layers. (default: ``2``)
        act (str, optional): Activation function. (default: ``"gelu"``)
        pooling (str, optional): Graph pooling method ('mean', 'add', 'max'). (default: ``"mean"``)
        **kwargs: Additional layer arguments.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int = 1,
        L: int = 2,
        act: str = "gelu",
        pooling: str = "mean",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.L = L
        self.act_name = act
        self.pooling = pooling.lower()

        self.FC_layers = []
        for l in range(L):
            in_dim_l = dim_in // (2**l)
            out_dim_l = dim_in // (2 ** (l + 1))
            self.FC_layers.append(
                layers.Dense(out_dim_l, use_bias=True, name=f"FC_layers_{l}")
            )
        self.FC_layers.append(
            layers.Dense(dim_out, use_bias=True, name=f"FC_layers_{L}")
        )

    def build(self, input_shape=None):
        curr_dim = self.dim_in
        for layer in self.FC_layers:
            if not layer.built:
                layer.build((None, curr_dim))
            curr_dim = layer.units
        super().build(input_shape)

    def call(self, x, batch=None, training=False):
        if self.pooling in ["mean", "avg"]:
            graph_emb = global_mean_pool(x, batch)
        elif self.pooling in ["add", "sum"]:
            graph_emb = global_add_pool(x, batch)
        elif self.pooling == "max":
            graph_emb = global_max_pool(x, batch)
        else:
            raise ValueError(f"Unknown pooling method '{self.pooling}'")

        act_fn = _get_act(self.act_name)
        for l in range(self.L):
            graph_emb = self.FC_layers[l](graph_emb)
            graph_emb = act_fn(graph_emb)

        graph_emb = self.FC_layers[self.L](graph_emb)
        return graph_emb


class GPSModel(keras.Model):
    r"""GraphGPS: General Powerful Scalable Graph Transformer from the
    `"Recipe for a General, Powerful, Scalable Graph Transformer"
    <https://arxiv.org/abs/2205.12454>`_ paper (NeurIPS 2022).

    Args:
        dim_in (int, optional): Initial input feature dimension. (default: ``256``)
        dim_out (int, optional): Target output dimension. (default: ``1``)
        num_layers (int, optional): Number of GPS layers. (default: ``16``)
        dim_hidden (int, optional): Hidden embedding dimension. (default: ``256``)
        num_heads (int, optional): Number of attention heads. (default: ``8``)
        local_gnn_type (str, optional): Local MPNN layer type. (default: ``"CustomGatedGCN"``)
        act (str, optional): Activation function. (default: ``"gelu"``)
        dropout (float, optional): Dropout probability. (default: ``0.1``)
        attn_dropout (float, optional): Attention dropout probability. (default: ``0.1``)
        batch_norm (bool, optional): Whether to use batch normalization. (default: ``True``)
        layer_norm (bool, optional): Whether to use layer normalization. (default: ``False``)
        node_encoder_type (str, optional): Node encoder type ("Atom+RWSE", "Atom", "Linear", or None). (default: ``"Atom+RWSE"``)
        edge_encoder_type (str, optional): Edge encoder type ("Bond", "Linear", or None). (default: ``"Bond"``)
        atom_feature_dims (List[int], optional): Categorical feature vocabulary sizes for atom features.
        bond_feature_dims (List[int], optional): Categorical feature vocabulary sizes for bond features.
        rwse_num_steps (int, optional): Number of RWSE steps. (default: ``16``)
        rwse_dim_pe (int, optional): RWSE embedding dimension. (default: ``20``)
        graph_pooling (str, optional): Graph pooling type ('mean', 'add', 'max'). (default: ``"mean"``)
        head_layers (int, optional): Number of hidden layers in prediction head. (default: ``2``)
        **kwargs: Additional model arguments.
    """

    def __init__(
        self,
        dim_in: int = 256,
        dim_out: int = 1,
        num_layers: int = 16,
        dim_hidden: int = 256,
        num_heads: int = 8,
        local_gnn_type: str = "CustomGatedGCN",
        act: str = "gelu",
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        batch_norm: bool = True,
        layer_norm: bool = False,
        node_encoder_type: Optional[str] = "Atom+RWSE",
        edge_encoder_type: Optional[str] = "Bond",
        atom_feature_dims: Optional[List[int]] = None,
        bond_feature_dims: Optional[List[int]] = None,
        rwse_num_steps: int = 16,
        rwse_dim_pe: int = 20,
        graph_pooling: str = "mean",
        head_layers: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden
        self.num_heads = num_heads
        self.local_gnn_type = local_gnn_type
        self.act_name = act
        self.dropout_rate = dropout
        self.attn_dropout_rate = attn_dropout
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        self.node_encoder_type = node_encoder_type
        self.edge_encoder_type = edge_encoder_type
        self.rwse_num_steps = rwse_num_steps
        self.rwse_dim_pe = rwse_dim_pe
        self.graph_pooling = graph_pooling
        self.head_layers = head_layers

        # Node Encoder
        if node_encoder_type == "Atom+RWSE":
            self.atom_encoder = AtomEncoder(
                emb_dim=dim_hidden - rwse_dim_pe,
                feature_dims=atom_feature_dims,
                name="atom_encoder",
            )
            self.rwse_encoder = RWSEEncoder(
                num_rw_steps=rwse_num_steps,
                pe_dim=rwse_dim_pe,
                name="rwse_encoder",
            )
        elif node_encoder_type == "Atom":
            self.atom_encoder = AtomEncoder(
                emb_dim=dim_hidden,
                feature_dims=atom_feature_dims,
                name="atom_encoder",
            )
            self.rwse_encoder = None
        elif node_encoder_type == "Linear":
            self.linear_node_encoder = layers.Dense(dim_hidden, name="linear_node_encoder")
            self.atom_encoder = None
            self.rwse_encoder = None
        else:
            self.atom_encoder = None
            self.rwse_encoder = None
            self.linear_node_encoder = None

        # Edge Encoder
        if edge_encoder_type == "Bond":
            self.bond_encoder = BondEncoder(
                emb_dim=dim_hidden,
                feature_dims=bond_feature_dims,
                name="bond_encoder",
            )
        elif edge_encoder_type == "Linear":
            self.linear_edge_encoder = layers.Dense(dim_hidden, name="linear_edge_encoder")
            self.bond_encoder = None
        else:
            self.bond_encoder = None
            self.linear_edge_encoder = None

        # GPS Layers
        self.gps_layers = [
            GPSLayer(
                dim_h=dim_hidden,
                local_gnn_type=local_gnn_type,
                global_model_type="Transformer",
                num_heads=num_heads,
                act=act,
                dropout=dropout,
                attn_dropout=attn_dropout,
                layer_norm=layer_norm,
                batch_norm=batch_norm,
                name=f"gps_layer_{i}",
            )
            for i in range(num_layers)
        ]

        # SANGraphHead
        self.post_mp = SANGraphHead(
            dim_in=dim_hidden,
            dim_out=dim_out,
            L=head_layers,
            act=act,
            pooling=graph_pooling,
            name="post_mp",
        )

    def build(self, input_shape=None):
        if self.atom_encoder is not None and not self.atom_encoder.built:
            self.atom_encoder.build(None)
        if self.rwse_encoder is not None and not self.rwse_encoder.built:
            self.rwse_encoder.build(None)
        if self.bond_encoder is not None and not self.bond_encoder.built:
            self.bond_encoder.build(None)
        for layer in self.gps_layers:
            if not layer.built:
                layer.build(None)
        if not self.post_mp.built:
            self.post_mp.build(None)
        super().build(input_shape)

    def call(
        self,
        x,
        edge_index,
        edge_attr=None,
        pestat_RWSE=None,
        batch=None,
        training=False,
    ):
        # Node encoding
        if self.node_encoder_type == "Atom+RWSE":
            x_emb = self.atom_encoder(x)
            if pestat_RWSE is not None and self.rwse_encoder is not None:
                pe_emb = self.rwse_encoder(pestat_RWSE, training=training)
                x = ops.concatenate([x_emb, pe_emb], axis=-1)
            else:
                x = x_emb
        elif self.node_encoder_type == "Atom":
            x = self.atom_encoder(x)
        elif self.node_encoder_type == "Linear" and self.linear_node_encoder is not None:
            x = self.linear_node_encoder(x)

        # Edge encoding
        if self.edge_encoder_type == "Bond" and edge_attr is not None:
            edge_attr = self.bond_encoder(edge_attr)
        elif self.edge_encoder_type == "Linear" and self.linear_edge_encoder is not None:
            edge_attr = self.linear_edge_encoder(edge_attr)

        # GPS layers
        for layer in self.gps_layers:
            x, edge_attr = layer(
                x, edge_index, edge_attr, batch=batch, training=training
            )

        # Head
        pred = self.post_mp(x, batch=batch, training=training)
        return pred


def load_gps_weights(model: GPSModel, checkpoint_path: str):
    r"""Loads trained PyTorch GraphGPS checkpoint weights into a Keras 3 `GPSModel`.

    Args:
        model (GPSModel): Target `GPSModel` instance.
        checkpoint_path (str): Path to PyTorch `.ckpt` or `.pt` checkpoint file.

    Returns:
        GPSModel: The model with loaded weights.
    """
    import torch

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    clean_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            k = k[6:]
        clean_dict[k] = v

    def _to_tensor(t):
        arr = t.detach().cpu().numpy()
        return ops.convert_to_tensor(arr, dtype="float32")

    # Build model if needed
    if not model.built:
        model.build(None)

    # 1. Node Encoder
    if model.node_encoder_type in ["Atom+RWSE", "Atom"] and model.atom_encoder is not None:
        for i, emb in enumerate(model.atom_encoder.atom_embedding_list):
            k = f"encoder.node_encoder.encoder1.atom_embedding_list.{i}.weight"
            if k in clean_dict:
                emb.embeddings.assign(_to_tensor(clean_dict[k]))

    if model.node_encoder_type == "Atom+RWSE" and model.rwse_encoder is not None:
        p = "encoder.node_encoder.encoder2"
        if f"{p}.raw_norm.weight" in clean_dict:
            model.rwse_encoder.raw_norm.gamma.assign(_to_tensor(clean_dict[f"{p}.raw_norm.weight"]))
        if f"{p}.raw_norm.bias" in clean_dict:
            model.rwse_encoder.raw_norm.beta.assign(_to_tensor(clean_dict[f"{p}.raw_norm.bias"]))
        if f"{p}.raw_norm.running_mean" in clean_dict:
            model.rwse_encoder.raw_norm.moving_mean.assign(_to_tensor(clean_dict[f"{p}.raw_norm.running_mean"]))
        if f"{p}.raw_norm.running_var" in clean_dict:
            model.rwse_encoder.raw_norm.moving_variance.assign(_to_tensor(clean_dict[f"{p}.raw_norm.running_var"]))

        if f"{p}.pe_encoder.weight" in clean_dict:
            model.rwse_encoder.pe_encoder.kernel.assign(_to_tensor(clean_dict[f"{p}.pe_encoder.weight"].t()))
        if f"{p}.pe_encoder.bias" in clean_dict:
            model.rwse_encoder.pe_encoder.bias.assign(_to_tensor(clean_dict[f"{p}.pe_encoder.bias"]))

    # 2. Edge Encoder
    if model.edge_encoder_type == "Bond" and model.bond_encoder is not None:
        for i, emb in enumerate(model.bond_encoder.bond_embedding_list):
            k = f"encoder.edge_encoder.bond_embedding_list.{i}.weight"
            if k in clean_dict:
                emb.embeddings.assign(_to_tensor(clean_dict[k]))

    # 3. GPS Layers
    dim_h = model.dim_hidden
    for i, gps_layer in enumerate(model.gps_layers):
        p = f"layers.{i}"

        # Local MPNN: CustomGatedGCN
        if gps_layer.local_model is not None:
            lm = gps_layer.local_model
            for proj_name in ["A", "B", "C", "D", "E"]:
                dense_proj = getattr(lm, proj_name)
                if f"{p}.local_model.{proj_name}.weight" in clean_dict:
                    dense_proj.kernel.assign(_to_tensor(clean_dict[f"{p}.local_model.{proj_name}.weight"].t()))
                if f"{p}.local_model.{proj_name}.bias" in clean_dict:
                    dense_proj.bias.assign(_to_tensor(clean_dict[f"{p}.local_model.{proj_name}.bias"]))

            for bn_name, bn_layer in [("bn_node_x", lm.bn_node_x), ("bn_edge_e", lm.bn_edge_e)]:
                if f"{p}.local_model.{bn_name}.weight" in clean_dict:
                    bn_layer.gamma.assign(_to_tensor(clean_dict[f"{p}.local_model.{bn_name}.weight"]))
                if f"{p}.local_model.{bn_name}.bias" in clean_dict:
                    bn_layer.beta.assign(_to_tensor(clean_dict[f"{p}.local_model.{bn_name}.bias"]))
                if f"{p}.local_model.{bn_name}.running_mean" in clean_dict:
                    bn_layer.moving_mean.assign(_to_tensor(clean_dict[f"{p}.local_model.{bn_name}.running_mean"]))
                if f"{p}.local_model.{bn_name}.running_var" in clean_dict:
                    bn_layer.moving_variance.assign(_to_tensor(clean_dict[f"{p}.local_model.{bn_name}.running_var"]))

            if gps_layer.norm1_local is not None:
                if f"{p}.norm1_local.weight" in clean_dict:
                    gps_layer.norm1_local.gamma.assign(_to_tensor(clean_dict[f"{p}.norm1_local.weight"]))
                if f"{p}.norm1_local.bias" in clean_dict:
                    gps_layer.norm1_local.beta.assign(_to_tensor(clean_dict[f"{p}.norm1_local.bias"]))
                if hasattr(gps_layer.norm1_local, "moving_mean") and f"{p}.norm1_local.running_mean" in clean_dict:
                    gps_layer.norm1_local.moving_mean.assign(_to_tensor(clean_dict[f"{p}.norm1_local.running_mean"]))
                if hasattr(gps_layer.norm1_local, "moving_variance") and f"{p}.norm1_local.running_var" in clean_dict:
                    gps_layer.norm1_local.moving_variance.assign(_to_tensor(clean_dict[f"{p}.norm1_local.running_var"]))

        # Global Attention: MultiHeadAttention
        if gps_layer.self_attn is not None:
            sa = gps_layer.self_attn
            if f"{p}.self_attn.in_proj_weight" in clean_dict:
                in_w = clean_dict[f"{p}.self_attn.in_proj_weight"]
                sa.q_proj.kernel.assign(_to_tensor(in_w[:dim_h, :].t()))
                sa.k_proj.kernel.assign(_to_tensor(in_w[dim_h:2*dim_h, :].t()))
                sa.v_proj.kernel.assign(_to_tensor(in_w[2*dim_h:, :].t()))

            if f"{p}.self_attn.in_proj_bias" in clean_dict:
                in_b = clean_dict[f"{p}.self_attn.in_proj_bias"]
                sa.q_proj.bias.assign(_to_tensor(in_b[:dim_h]))
                sa.k_proj.bias.assign(_to_tensor(in_b[dim_h:2*dim_h]))
                sa.v_proj.bias.assign(_to_tensor(in_b[2*dim_h:]))

            if f"{p}.self_attn.out_proj.weight" in clean_dict:
                sa.out_proj.kernel.assign(_to_tensor(clean_dict[f"{p}.self_attn.out_proj.weight"].t()))
            if f"{p}.self_attn.out_proj.bias" in clean_dict:
                sa.out_proj.bias.assign(_to_tensor(clean_dict[f"{p}.self_attn.out_proj.bias"]))

            if gps_layer.norm1_attn is not None:
                if f"{p}.norm1_attn.weight" in clean_dict:
                    gps_layer.norm1_attn.gamma.assign(_to_tensor(clean_dict[f"{p}.norm1_attn.weight"]))
                if f"{p}.norm1_attn.bias" in clean_dict:
                    gps_layer.norm1_attn.beta.assign(_to_tensor(clean_dict[f"{p}.norm1_attn.bias"]))
                if hasattr(gps_layer.norm1_attn, "moving_mean") and f"{p}.norm1_attn.running_mean" in clean_dict:
                    gps_layer.norm1_attn.moving_mean.assign(_to_tensor(clean_dict[f"{p}.norm1_attn.running_mean"]))
                if hasattr(gps_layer.norm1_attn, "moving_variance") and f"{p}.norm1_attn.running_var" in clean_dict:
                    gps_layer.norm1_attn.moving_variance.assign(_to_tensor(clean_dict[f"{p}.norm1_attn.running_var"]))

        # FFN
        if f"{p}.ff_linear1.weight" in clean_dict:
            gps_layer.ff_linear1.kernel.assign(_to_tensor(clean_dict[f"{p}.ff_linear1.weight"].t()))
        if f"{p}.ff_linear1.bias" in clean_dict:
            gps_layer.ff_linear1.bias.assign(_to_tensor(clean_dict[f"{p}.ff_linear1.bias"]))
        if f"{p}.ff_linear2.weight" in clean_dict:
            gps_layer.ff_linear2.kernel.assign(_to_tensor(clean_dict[f"{p}.ff_linear2.weight"].t()))
        if f"{p}.ff_linear2.bias" in clean_dict:
            gps_layer.ff_linear2.bias.assign(_to_tensor(clean_dict[f"{p}.ff_linear2.bias"]))

        if gps_layer.norm2 is not None:
            if f"{p}.norm2.weight" in clean_dict:
                gps_layer.norm2.gamma.assign(_to_tensor(clean_dict[f"{p}.norm2.weight"]))
            if f"{p}.norm2.bias" in clean_dict:
                gps_layer.norm2.beta.assign(_to_tensor(clean_dict[f"{p}.norm2.bias"]))
            if hasattr(gps_layer.norm2, "moving_mean") and f"{p}.norm2.running_mean" in clean_dict:
                gps_layer.norm2.moving_mean.assign(_to_tensor(clean_dict[f"{p}.norm2.running_mean"]))
            if hasattr(gps_layer.norm2, "moving_variance") and f"{p}.norm2.running_var" in clean_dict:
                gps_layer.norm2.moving_variance.assign(_to_tensor(clean_dict[f"{p}.norm2.running_var"]))
            if hasattr(gps_layer.norm2, "moving_mean") and f"{p}.norm2.running_mean" in clean_dict:
                gps_layer.norm2.moving_mean.assign(_to_tensor(clean_dict[f"{p}.norm2.running_mean"]))
            if hasattr(gps_layer.norm2, "moving_variance") and f"{p}.norm2.running_var" in clean_dict:
                gps_layer.norm2.moving_variance.assign(_to_tensor(clean_dict[f"{p}.norm2.running_var"]))

    # 4. SANGraphHead (post_mp)
    for l, fc in enumerate(model.post_mp.FC_layers):
        if f"post_mp.FC_layers.{l}.weight" in clean_dict:
            fc.kernel.assign(_to_tensor(clean_dict[f"post_mp.FC_layers.{l}.weight"].t()))
        if f"post_mp.FC_layers.{l}.bias" in clean_dict:
            fc.bias.assign(_to_tensor(clean_dict[f"post_mp.FC_layers.{l}.bias"]))

    return model


DROPBOX_CHECKPOINT_URLS = {
    "pcqm4m-GPS+RWSE.deep": "https://www.dropbox.com/s/aomimvak4gb6et3/pcqm4m-GPS%2BRWSE.deep.zip?dl=1",
}


def download_gps_checkpoint(
    checkpoint_name: str = "pcqm4m-GPS+RWSE.deep",
    cache_dir: Optional[str] = None,
) -> str:
    r"""Downloads and extracts a pretrained GraphGPS checkpoint.

    Args:
        checkpoint_name (str, optional): Name of the checkpoint.
            Currently supported: ``"pcqm4m-GPS+RWSE.deep"``.
        cache_dir (str, optional): Cache directory to store downloaded checkpoint.

    Returns:
        str: Absolute path to the extracted `.ckpt` checkpoint file.
    """
    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.cache/k3_node/graphgps")
    os.makedirs(cache_dir, exist_ok=True)

    expected_ckpt = os.path.join(cache_dir, checkpoint_name, "0", "ckpt", "148.ckpt")
    if os.path.exists(expected_ckpt):
        return expected_ckpt

    # Also check repo pretrained dir if available
    local_repo_ckpt = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "..",
        "GraphGPS",
        "pretrained",
        checkpoint_name,
        "0",
        "ckpt",
        "148.ckpt",
    )
    if os.path.exists(local_repo_ckpt):
        return os.path.abspath(local_repo_ckpt)

    url = DROPBOX_CHECKPOINT_URLS.get(checkpoint_name)
    if url is None:
        raise ValueError(f"Unknown checkpoint '{checkpoint_name}'")

    zip_path = os.path.join(cache_dir, f"{checkpoint_name}.zip")
    if not os.path.exists(zip_path):
        print(f"Downloading {checkpoint_name} from {url}...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp, open(zip_path, "wb") as f:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(cache_dir)

    if not os.path.exists(expected_ckpt):
        # Search for any ckpt file inside extracted directory
        for root, _, files in os.walk(cache_dir):
            for file in files:
                if file.endswith(".ckpt"):
                    return os.path.join(root, file)
        raise FileNotFoundError(f"Could not find .ckpt inside extracted {zip_path}")

    return expected_ckpt
