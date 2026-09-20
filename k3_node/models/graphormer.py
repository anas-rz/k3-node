import math
import os
import urllib.request
from typing import Optional, Union, Tuple, List, Dict, Any

import keras
from keras import layers, ops


class GraphNodeFeature(layers.Layer):
    r"""Computes initial node representations by summing atom feature embeddings,
    in-degree embeddings, out-degree embeddings, and prepending a learnable graph token.

    Args:
        num_atoms (int): Maximum number of atom types.
        num_in_degree (int): Maximum in-degree value.
        num_out_degree (int): Maximum out-degree value.
        hidden_dim (int): Embedding dimension.
        **kwargs: Additional layer arguments.
    """

    def __init__(
        self,
        num_atoms: int,
        num_in_degree: int,
        num_out_degree: int,
        hidden_dim: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_atoms = num_atoms
        self.num_in_degree = num_in_degree
        self.num_out_degree = num_out_degree
        self.hidden_dim = hidden_dim

        # 1-indexed padding_idx=0 in fairseq / Graphormer
        self.atom_encoder = layers.Embedding(
            input_dim=num_atoms + 1,
            output_dim=hidden_dim,
            mask_zero=False,
            name="atom_encoder",
        )
        self.in_degree_encoder = layers.Embedding(
            input_dim=num_in_degree,
            output_dim=hidden_dim,
            mask_zero=False,
            name="in_degree_encoder",
        )
        self.out_degree_encoder = layers.Embedding(
            input_dim=num_out_degree,
            output_dim=hidden_dim,
            mask_zero=False,
            name="out_degree_encoder",
        )
        self.graph_token = layers.Embedding(
            input_dim=1,
            output_dim=hidden_dim,
            name="graph_token",
        )

    def build(self, input_shape=None):
        if not self.built:
            self.atom_encoder.build(None)
            self.in_degree_encoder.build(None)
            self.out_degree_encoder.build(None)
            self.graph_token.build(None)
        super().build(input_shape)

    def call(self, x, in_degree, out_degree):
        r"""
        Args:
            x (Tensor): Atom features of shape ``[batch_size, num_nodes, num_features]``
                or ``[batch_size, num_nodes]``.
            in_degree (Tensor): In-degrees of shape ``[batch_size, num_nodes]``.
            out_degree (Tensor): Out-degrees of shape ``[batch_size, num_nodes]``.

        Returns:
            Tensor: Node features with prepended graph token of shape
            ``[batch_size, num_nodes + 1, hidden_dim]``.
        """
        x_shape = ops.shape(x)
        batch_size, num_nodes = x_shape[0], x_shape[1]

        if len(ops.shape(x)) == 2:
            x_emb = self.atom_encoder(x)
        else:
            # Multi-dimensional atom features: sum over feature dim
            x_emb = ops.sum(self.atom_encoder(x), axis=-2)

        node_feature = (
            x_emb
            + self.in_degree_encoder(in_degree)
            + self.out_degree_encoder(out_degree)
        )

        # Graph token feature: shape [batch_size, 1, hidden_dim]
        token_id = ops.zeros((batch_size, 1), dtype="int32")
        graph_token_feature = self.graph_token(token_id)

        graph_node_feature = ops.concatenate([graph_token_feature, node_feature], axis=1)
        return graph_node_feature


class GraphAttnBias(layers.Layer):
    r"""Computes the structural attention bias for each attention head from shortest path
    distances (spatial encoding) and edge features (edge encoding).

    Args:
        num_heads (int): Number of attention heads.
        num_atoms (int): Maximum number of atom types.
        num_edges (int): Maximum number of edge types.
        num_spatial (int): Maximum spatial distance.
        num_edge_dis (int): Maximum edge distance multiplier for multi-hop encoding.
        edge_type (str, optional): Type of edge encoding (``"multi_hop"`` or ``"single_hop"``).
            (default: ``"multi_hop"``)
        multi_hop_max_dist (int, optional): Maximum distance for multi-hop paths. (default: ``20``)
        **kwargs: Additional layer arguments.
    """

    def __init__(
        self,
        num_heads: int,
        num_atoms: int,
        num_edges: int,
        num_spatial: int,
        num_edge_dis: int,
        edge_type: str = "multi_hop",
        multi_hop_max_dist: int = 20,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.num_atoms = num_atoms
        self.num_edges = num_edges
        self.num_spatial = num_spatial
        self.num_edge_dis = num_edge_dis
        self.edge_type = edge_type
        self.multi_hop_max_dist = multi_hop_max_dist

        self.edge_encoder = layers.Embedding(
            input_dim=num_edges + 1,
            output_dim=num_heads,
            name="edge_encoder",
        )
        if self.edge_type == "multi_hop":
            self.edge_dis_encoder = layers.Embedding(
                input_dim=num_edge_dis * num_heads * num_heads,
                output_dim=1,
                name="edge_dis_encoder",
            )
        self.spatial_pos_encoder = layers.Embedding(
            input_dim=num_spatial,
            output_dim=num_heads,
            name="spatial_pos_encoder",
        )
        self.graph_token_virtual_distance = layers.Embedding(
            input_dim=1,
            output_dim=num_heads,
            name="graph_token_virtual_distance",
        )

    def build(self, input_shape=None):
        if not self.built:
            self.edge_encoder.build(None)
            self.spatial_pos_encoder.build(None)
            self.graph_token_virtual_distance.build(None)
            if hasattr(self, "edge_dis_encoder"):
                self.edge_dis_encoder.build(None)
        super().build(input_shape)

    def call(
        self,
        attn_bias,
        spatial_pos,
        x,
        edge_input=None,
        attn_edge_type=None,
    ):
        r"""
        Args:
            attn_bias (Tensor): Base attention mask of shape ``[batch_size, num_nodes + 1, num_nodes + 1]``.
            spatial_pos (Tensor): Shortest path matrix of shape ``[batch_size, num_nodes, num_nodes]``.
            x (Tensor): Atom features of shape ``[batch_size, num_nodes, ...]``.
            edge_input (Tensor, optional): Multi-hop edge features along shortest paths of shape
                ``[batch_size, num_nodes, num_nodes, max_dist, edge_feat_dim]``.
            attn_edge_type (Tensor, optional): Single-hop edge types of shape
                ``[batch_size, num_nodes, num_nodes, edge_feat_dim]``.

        Returns:
            Tensor: Attention bias tensor of shape ``[batch_size, num_heads, num_nodes + 1, num_nodes + 1]``.
        """
        x_shape = ops.shape(x)
        batch_size, num_nodes = x_shape[0], x_shape[1]

        # [batch_size, num_heads, num_nodes + 1, num_nodes + 1]
        graph_attn_bias = ops.repeat(
            ops.expand_dims(attn_bias, axis=1), repeats=self.num_heads, axis=1
        )

        # Spatial position bias: [batch_size, num_nodes, num_nodes, num_heads] -> [batch_size, num_heads, num_nodes, num_nodes]
        spatial_pos_bias = ops.transpose(self.spatial_pos_encoder(spatial_pos), (0, 3, 1, 2))

        # Update node-to-node submatrix [1:, 1:]
        sub_bias = graph_attn_bias[:, :, 1:, 1:] + spatial_pos_bias

        # Virtual distance bias for graph token: shape [1, num_heads, 1]
        t = ops.reshape(self.graph_token_virtual_distance(ops.zeros((1,), dtype="int32")), (1, self.num_heads, 1, 1))
        # Add to row 0 (graph token to all nodes) and column 0 (all nodes to graph token)
        row0 = graph_attn_bias[:, :, 0:1, 1:] + t[:, :, :, 0:]
        col0 = graph_attn_bias[:, :, 1:, 0:1] + t[:, :, 0:, :]
        corner = graph_attn_bias[:, :, 0:1, 0:1] + t

        # Edge feature bias
        if self.edge_type == "multi_hop" and edge_input is not None:
            spatial_pos_ = ops.copy(spatial_pos)
            # Replace 0 with 1 for padding
            spatial_pos_ = ops.where(ops.equal(spatial_pos_, 0), 1, spatial_pos_)
            spatial_pos_ = ops.where(spatial_pos_ > 1, spatial_pos_ - 1, spatial_pos_)
            if self.multi_hop_max_dist > 0:
                spatial_pos_ = ops.clip(spatial_pos_, 0, self.multi_hop_max_dist)
                edge_input = edge_input[:, :, :, : self.multi_hop_max_dist, :]

            # edge_input: [batch_size, num_nodes, num_nodes, max_dist, edge_feat_dim]
            # edge_encoder -> [batch_size, num_nodes, num_nodes, max_dist, edge_feat_dim, num_heads]
            edge_enc = ops.mean(self.edge_encoder(edge_input), axis=-2)
            max_dist = ops.shape(edge_enc)[3]

            # edge_enc: [batch_size, num_nodes, num_nodes, max_dist, num_heads]
            # permute to [max_dist, batch_size * num_nodes * num_nodes, num_heads]
            edge_enc_perm = ops.transpose(edge_enc, (3, 0, 1, 2, 4))
            edge_input_flat = ops.reshape(edge_enc_perm, (max_dist, -1, self.num_heads))

            # Weight shape for edge_dis_encoder: [num_edge_dis * num_heads * num_heads, 1]
            if not self.edge_dis_encoder.built:
                self.edge_dis_encoder.build(None)
            dis_weights = ops.reshape(
                self.edge_dis_encoder.weights[0], (-1, self.num_heads, self.num_heads)
            )[:max_dist, :, :]

            edge_input_flat = ops.matmul(edge_input_flat, dis_weights)
            edge_enc_back = ops.reshape(
                edge_input_flat, (max_dist, batch_size, num_nodes, num_nodes, self.num_heads)
            )
            # Permute to [batch_size, num_nodes, num_nodes, max_dist, num_heads]
            edge_enc_back = ops.transpose(edge_enc_back, (1, 2, 3, 0, 4))

            # Sum over distance dim and divide by path length:
            sp_float = ops.expand_dims(ops.cast(spatial_pos_, "float32"), axis=-1)
            edge_bias = ops.sum(edge_enc_back, axis=3) / sp_float
            edge_bias = ops.transpose(edge_bias, (0, 3, 1, 2))
            sub_bias = sub_bias + edge_bias
        elif attn_edge_type is not None:
            edge_bias = ops.mean(self.edge_encoder(attn_edge_type), axis=-2)
            edge_bias = ops.transpose(edge_bias, (0, 3, 1, 2))
            sub_bias = sub_bias + edge_bias

        # Reconstruct graph_attn_bias:
        top_row = ops.concatenate([corner, row0], axis=3)
        bottom_rows = ops.concatenate([col0, sub_bias], axis=3)
        graph_attn_bias = ops.concatenate([top_row, bottom_rows], axis=2)

        # Reset padding elements with -inf mask
        graph_attn_bias = graph_attn_bias + ops.expand_dims(attn_bias, axis=1)
        return graph_attn_bias


class GraphormerMultiheadAttention(layers.Layer):
    r"""Multi-head self-attention layer with support for additive graph structural attention bias
    and key padding masks.

    Args:
        embed_dim (int): Total embedding dimension.
        num_heads (int): Number of attention heads.
        dropout (float, optional): Attention dropout probability. (default: ``0.0``)
        bias (bool, optional): Whether to use bias in linear projections. (default: ``True``)
        **kwargs: Additional layer arguments.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != embed_dim:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads}).")

        self.scaling = self.head_dim ** -0.5

        self.q_proj = layers.Dense(embed_dim, use_bias=bias, name="q_proj")
        self.k_proj = layers.Dense(embed_dim, use_bias=bias, name="k_proj")
        self.v_proj = layers.Dense(embed_dim, use_bias=bias, name="v_proj")
        self.out_proj = layers.Dense(embed_dim, use_bias=bias, name="out_proj")
        self.dropout = layers.Dropout(dropout)

    def build(self, input_shape=None):
        if not self.built:
            self.q_proj.build((None, None, self.embed_dim))
            self.k_proj.build((None, None, self.embed_dim))
            self.v_proj.build((None, None, self.embed_dim))
            self.out_proj.build((None, None, self.embed_dim))
        super().build(input_shape)

    def call(
        self,
        x,
        attn_bias=None,
        key_padding_mask=None,
        training: bool = False,
    ):
        r"""
        Args:
            x (Tensor): Sequence representation of shape ``[batch_size, seq_len, embed_dim]``.
            attn_bias (Tensor, optional): Additive bias of shape
                ``[batch_size, num_heads, seq_len, seq_len]``.
            key_padding_mask (Tensor, optional): Boolean padding mask of shape ``[batch_size, seq_len]``,
                where True indicates padding tokens to ignore.
            training (bool, optional): Whether in training mode. (default: ``False``)

        Returns:
            Tuple[Tensor, Tensor]: Output tensor of shape ``[batch_size, seq_len, embed_dim]``
            and attention weights of shape ``[batch_size, num_heads, seq_len, seq_len]``.
        """
        shape = ops.shape(x)
        batch_size, seq_len = shape[0], shape[1]

        q = self.q_proj(x) * self.scaling
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to [batch_size, num_heads, seq_len, head_dim]
        q = ops.transpose(ops.reshape(q, (batch_size, seq_len, self.num_heads, self.head_dim)), (0, 2, 1, 3))
        k = ops.transpose(ops.reshape(k, (batch_size, seq_len, self.num_heads, self.head_dim)), (0, 2, 1, 3))
        v = ops.transpose(ops.reshape(v, (batch_size, seq_len, self.num_heads, self.head_dim)), (0, 2, 1, 3))

        # [batch_size, num_heads, seq_len, seq_len]
        attn_weights = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2)))

        if attn_bias is not None:
            attn_weights = attn_weights + attn_bias

        if key_padding_mask is not None:
            # key_padding_mask: [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
            mask = ops.expand_dims(ops.expand_dims(key_padding_mask, axis=1), axis=2)
            attn_weights = ops.where(mask, -1e9, attn_weights)

        attn_probs = ops.softmax(attn_weights, axis=-1)
        attn_probs = self.dropout(attn_probs, training=training)

        # [batch_size, num_heads, seq_len, head_dim]
        attn = ops.matmul(attn_probs, v)

        # Reshape to [batch_size, seq_len, embed_dim]
        attn = ops.reshape(ops.transpose(attn, (0, 2, 1, 3)), (batch_size, seq_len, self.embed_dim))
        out = self.out_proj(attn)
        return out, attn_probs


class GraphormerGraphEncoderLayer(layers.Layer):
    r"""A single Graphormer Transformer Encoder Layer, supporting Pre-LN or Post-LN,
    multi-head attention with structural attention bias, and a 2-layer FFN.

    Args:
        embedding_dim (int, optional): Embedding dimension. (default: ``768``)
        ffn_embedding_dim (int, optional): FFN intermediate dimension. (default: ``768``)
        num_attention_heads (int, optional): Number of attention heads. (default: ``32``)
        dropout (float, optional): Dropout probability. (default: ``0.1``)
        attention_dropout (float, optional): Attention dropout. (default: ``0.1``)
        activation_dropout (float, optional): FFN activation dropout. (default: ``0.1``)
        activation_fn (str, optional): Activation function (``"gelu"`` or ``"relu"``). (default: ``"gelu"``)
        pre_layernorm (bool, optional): Whether to use Pre-LN. (default: ``False``)
        **kwargs: Additional layer arguments.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 768,
        num_attention_heads: int = 32,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "gelu",
        pre_layernorm: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.ffn_embedding_dim = ffn_embedding_dim
        self.num_attention_heads = num_attention_heads
        self.dropout_rate = dropout
        self.pre_layernorm = pre_layernorm

        self.self_attn = GraphormerMultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_attention_heads,
            dropout=attention_dropout,
            name="self_attn",
        )
        self.self_attn_layer_norm = layers.LayerNormalization(
            epsilon=1e-5, name="self_attn_layer_norm"
        )
        self.dropout = layers.Dropout(dropout)

        self.fc1 = layers.Dense(ffn_embedding_dim, name="fc1")
        self.fc2 = layers.Dense(embedding_dim, name="fc2")
        self.act_dropout = layers.Dropout(activation_dropout)
        self.final_layer_norm = layers.LayerNormalization(
            epsilon=1e-5, name="final_layer_norm"
        )

        if activation_fn == "gelu":
            self.activation = ops.gelu
        elif activation_fn == "relu":
            self.activation = ops.relu
        else:
            self.activation = keras.activations.get(activation_fn)

    def build(self, input_shape=None):
        if not self.built:
            self.self_attn.build((None, None, self.embedding_dim))
            self.self_attn_layer_norm.build((None, None, self.embedding_dim))
            self.fc1.build((None, None, self.embedding_dim))
            self.fc2.build((None, None, self.ffn_embedding_dim))
            self.final_layer_norm.build((None, None, self.embedding_dim))
        super().build(input_shape)

    def call(
        self,
        x,
        attn_bias=None,
        key_padding_mask=None,
        training: bool = False,
    ):
        residual = x
        if self.pre_layernorm:
            x = self.self_attn_layer_norm(x)

        x, _ = self.self_attn(
            x,
            attn_bias=attn_bias,
            key_padding_mask=key_padding_mask,
            training=training,
        )
        x = self.dropout(x, training=training)
        x = residual + x

        if not self.pre_layernorm:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.pre_layernorm:
            x = self.final_layer_norm(x)

        x = self.fc2(self.act_dropout(self.activation(self.fc1(x)), training=training))
        x = self.dropout(x, training=training)
        x = residual + x

        if not self.pre_layernorm:
            x = self.final_layer_norm(x)

        return x


class GraphormerGraphEncoder(layers.Layer):
    r"""Graphormer Graph Encoder stack consisting of graph node features, structural attention bias,
    and multiple stacked encoder layers.

    Args:
        num_atoms (int): Maximum number of atom types.
        num_in_degree (int): Maximum in-degree value.
        num_out_degree (int): Maximum out-degree value.
        num_edges (int): Maximum number of edge types.
        num_spatial (int): Maximum spatial distance.
        num_edge_dis (int): Maximum edge distance multiplier.
        edge_type (str, optional): Edge encoding mode (``"multi_hop"`` or ``"single_hop"``). (default: ``"multi_hop"``)
        multi_hop_max_dist (int, optional): Max distance for multi-hop. (default: ``20``)
        num_encoder_layers (int, optional): Number of encoder layers. (default: ``12``)
        embedding_dim (int, optional): Embedding dimension. (default: ``768``)
        ffn_embedding_dim (int, optional): FFN intermediate dimension. (default: ``768``)
        num_attention_heads (int, optional): Number of attention heads. (default: ``32``)
        dropout (float, optional): Dropout probability. (default: ``0.1``)
        attention_dropout (float, optional): Attention dropout. (default: ``0.1``)
        activation_dropout (float, optional): FFN activation dropout. (default: ``0.1``)
        pre_layernorm (bool, optional): Whether to use Pre-LN. (default: ``False``)
        encoder_normalize_before (bool, optional): Whether to normalize before encoder blocks. (default: ``False``)
        activation_fn (str, optional): Activation function name. (default: ``"gelu"``)
        **kwargs: Additional layer arguments.
    """

    def __init__(
        self,
        num_atoms: int = 512,
        num_in_degree: int = 512,
        num_out_degree: int = 512,
        num_edges: int = 512,
        num_spatial: int = 512,
        num_edge_dis: int = 128,
        edge_type: str = "multi_hop",
        multi_hop_max_dist: int = 20,
        num_encoder_layers: int = 12,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 768,
        num_attention_heads: int = 32,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        pre_layernorm: bool = False,
        encoder_normalize_before: bool = False,
        activation_fn: str = "gelu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_encoder_layers = num_encoder_layers
        self.pre_layernorm = pre_layernorm

        self.graph_node_feature = GraphNodeFeature(
            num_atoms=num_atoms,
            num_in_degree=num_in_degree,
            num_out_degree=num_out_degree,
            hidden_dim=embedding_dim,
            name="graph_node_feature",
        )
        self.graph_attn_bias = GraphAttnBias(
            num_heads=num_attention_heads,
            num_atoms=num_atoms,
            num_edges=num_edges,
            num_spatial=num_spatial,
            num_edge_dis=num_edge_dis,
            edge_type=edge_type,
            multi_hop_max_dist=multi_hop_max_dist,
            name="graph_attn_bias",
        )
        if encoder_normalize_before:
            self.emb_layer_norm = layers.LayerNormalization(epsilon=1e-5, name="emb_layer_norm")
        else:
            self.emb_layer_norm = None

        self.dropout = layers.Dropout(dropout)

        self.encoder_layers = [
            GraphormerGraphEncoderLayer(
                embedding_dim=embedding_dim,
                ffn_embedding_dim=ffn_embedding_dim,
                num_attention_heads=num_attention_heads,
                dropout=dropout,
                attention_dropout=attention_dropout,
                activation_dropout=activation_dropout,
                activation_fn=activation_fn,
                pre_layernorm=pre_layernorm,
                name=f"layers_{i}",
            )
            for i in range(num_encoder_layers)
        ]

        if pre_layernorm:
            self.final_layer_norm = layers.LayerNormalization(epsilon=1e-5, name="final_layer_norm")
        else:
            self.final_layer_norm = None

    def build(self, input_shape=None):
        if not self.built:
            self.graph_node_feature.build(None)
            self.graph_attn_bias.build(None)
            if self.emb_layer_norm is not None:
                self.emb_layer_norm.build((None, None, self.embedding_dim))
            for layer in self.encoder_layers:
                layer.build((None, None, self.embedding_dim))
            if self.final_layer_norm is not None:
                self.final_layer_norm.build((None, None, self.embedding_dim))
        super().build(input_shape)

    def call(
        self,
        x,
        in_degree,
        out_degree,
        attn_bias,
        spatial_pos,
        edge_input=None,
        attn_edge_type=None,
        perturb=None,
        training: bool = False,
    ):
        batch_size = ops.shape(x)[0]
        # Compute padding mask: [batch_size, num_nodes]
        if len(ops.shape(x)) == 3:
            raw_mask = ops.equal(x[:, :, 0], 0)
        else:
            raw_mask = ops.equal(x, 0)

        # Prepend False for graph token: [batch_size, num_nodes + 1]
        cls_mask = ops.zeros((batch_size, 1), dtype="bool")
        padding_mask = ops.concatenate([cls_mask, raw_mask], axis=1)

        # Node features: [batch_size, num_nodes + 1, embedding_dim]
        h = self.graph_node_feature(x, in_degree, out_degree)
        if perturb is not None:
            # perturb is added to non-token nodes
            h_token = h[:, 0:1, :]
            h_nodes = h[:, 1:, :] + perturb
            h = ops.concatenate([h_token, h_nodes], axis=1)

        bias = self.graph_attn_bias(
            attn_bias=attn_bias,
            spatial_pos=spatial_pos,
            x=x,
            edge_input=edge_input,
            attn_edge_type=attn_edge_type,
        )

        if self.emb_layer_norm is not None:
            h = self.emb_layer_norm(h)

        h = self.dropout(h, training=training)

        for layer in self.encoder_layers:
            h = layer(
                h,
                attn_bias=bias,
                key_padding_mask=padding_mask,
                training=training,
            )

        if self.final_layer_norm is not None:
            h = self.final_layer_norm(h)

        graph_rep = h[:, 0, :]
        return h, graph_rep


class Graphormer(keras.Model):
    r"""Graphormer model for molecular graph representation and property prediction
    from `"Do Transformers Really Perform Badly for Graph Representation?" <https://arxiv.org/abs/2106.05234>`_.

    Args:
        num_atoms (int, optional): Maximum atom vocabulary size. (default: ``512``)
        num_in_degree (int, optional): Maximum in-degree. (default: ``512``)
        num_out_degree (int, optional): Maximum out-degree. (default: ``512``)
        num_edges (int, optional): Maximum edge vocabulary size. (default: ``512``)
        num_spatial (int, optional): Maximum spatial distance. (default: ``512``)
        num_edge_dis (int, optional): Maximum edge distance multiplier. (default: ``128``)
        edge_type (str, optional): Type of edge encoding (``"multi_hop"`` or ``"single_hop"``). (default: ``"multi_hop"``)
        multi_hop_max_dist (int, optional): Max distance for multi-hop paths. (default: ``20``)
        num_encoder_layers (int, optional): Number of Transformer layers. (default: ``12``)
        embedding_dim (int, optional): Hidden embedding dimension. (default: ``768``)
        ffn_embedding_dim (int, optional): FFN intermediate dimension. (default: ``768``)
        num_attention_heads (int, optional): Number of attention heads. (default: ``32``)
        dropout (float, optional): Dropout rate. (default: ``0.0``)
        attention_dropout (float, optional): Attention dropout rate. (default: ``0.1``)
        activation_dropout (float, optional): Activation dropout rate. (default: ``0.1``)
        encoder_normalize_before (bool, optional): Whether to apply LayerNorm before encoder blocks. (default: ``True``)
        pre_layernorm (bool, optional): Whether to use Pre-LN. (default: ``False``)
        num_classes (int, optional): Output dimension for prediction head. (default: ``1``)
        activation_fn (str, optional): Activation function. (default: ``"gelu"``)
        **kwargs: Additional model arguments.
    """

    def __init__(
        self,
        num_atoms: int = 512,
        num_in_degree: int = 512,
        num_out_degree: int = 512,
        num_edges: int = 512,
        num_spatial: int = 512,
        num_edge_dis: int = 128,
        edge_type: str = "multi_hop",
        multi_hop_max_dist: int = 20,
        num_encoder_layers: int = 12,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 768,
        num_attention_heads: int = 32,
        dropout: float = 0.0,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        encoder_normalize_before: bool = True,
        pre_layernorm: bool = False,
        num_classes: int = 1,
        activation_fn: str = "gelu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_atoms = num_atoms
        self.num_in_degree = num_in_degree
        self.num_out_degree = num_out_degree
        self.num_edges = num_edges
        self.num_spatial = num_spatial
        self.num_edge_dis = num_edge_dis
        self.edge_type = edge_type
        self.multi_hop_max_dist = multi_hop_max_dist
        self.num_encoder_layers = num_encoder_layers
        self.embedding_dim = embedding_dim
        self.ffn_embedding_dim = ffn_embedding_dim
        self.num_attention_heads = num_attention_heads
        self.num_classes = num_classes
        self.pre_layernorm = pre_layernorm

        self.graph_encoder = GraphormerGraphEncoder(
            num_atoms=num_atoms,
            num_in_degree=num_in_degree,
            num_out_degree=num_out_degree,
            num_edges=num_edges,
            num_spatial=num_spatial,
            num_edge_dis=num_edge_dis,
            edge_type=edge_type,
            multi_hop_max_dist=multi_hop_max_dist,
            num_encoder_layers=num_encoder_layers,
            embedding_dim=embedding_dim,
            ffn_embedding_dim=ffn_embedding_dim,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            encoder_normalize_before=encoder_normalize_before,
            pre_layernorm=pre_layernorm,
            activation_fn=activation_fn,
            name="graph_encoder",
        )

        # Output prediction head (matching GraphormerEncoder in Graphormer-main)
        self.lm_head_transform_weight = layers.Dense(
            embedding_dim, name="lm_head_transform_weight"
        )
        self.layer_norm = layers.LayerNormalization(epsilon=1e-5, name="layer_norm")
        self.embed_out = layers.Dense(num_classes, use_bias=False, name="embed_out")

        if activation_fn == "gelu":
            self.head_act = ops.gelu
        elif activation_fn == "relu":
            self.head_act = ops.relu
        else:
            self.head_act = keras.activations.get(activation_fn)

    def build(self, input_shape=None):
        if not self.built:
            self.graph_encoder.build(None)
            self.lm_head_transform_weight.build((None, self.embedding_dim))
            self.layer_norm.build((None, self.embedding_dim))
            self.embed_out.build((None, self.embedding_dim))
            self.lm_output_learned_bias = self.add_weight(
                shape=(1,),
                initializer="zeros",
                trainable=True,
                name="lm_output_learned_bias",
            )
        super().build(input_shape)

    def call(
        self,
        batched_data: Optional[Dict[str, Any]] = None,
        x=None,
        in_degree=None,
        out_degree=None,
        attn_bias=None,
        spatial_pos=None,
        edge_input=None,
        attn_edge_type=None,
        perturb=None,
        return_all: bool = False,
        training: bool = False,
    ):
        r"""Forward pass for Graphormer.

        Accepts either a single dictionary ``batched_data`` containing graph tensors,
        or individual tensor arguments.

        Returns:
            Tensor or Tuple[Tensor, Tensor]: Graph prediction tensor of shape ``[batch_size, num_classes]``,
            or if return_all=True, a tuple of ``(graph_pred, all_node_features)``.
        """
        if batched_data is not None:
            x = batched_data["x"]
            in_degree = batched_data["in_degree"]
            out_degree = batched_data["out_degree"]
            attn_bias = batched_data["attn_bias"]
            spatial_pos = batched_data["spatial_pos"]
            edge_input = batched_data.get("edge_input", None)
            attn_edge_type = batched_data.get("attn_edge_type", None)

        h, graph_rep = self.graph_encoder(
            x=x,
            in_degree=in_degree,
            out_degree=out_degree,
            attn_bias=attn_bias,
            spatial_pos=spatial_pos,
            edge_input=edge_input,
            attn_edge_type=attn_edge_type,
            perturb=perturb,
            training=training,
        )

        # Output projection on graph token: shape [batch_size, embedding_dim]
        token_h = self.layer_norm(self.head_act(self.lm_head_transform_weight(graph_rep)))
        out = self.embed_out(token_h)
        if hasattr(self, "lm_output_learned_bias") and self.lm_output_learned_bias is not None:
            out = out + self.lm_output_learned_bias

        if return_all:
            return out, h
        return out

    def embed(
        self,
        batched_data: Optional[Dict[str, Any]] = None,
        x=None,
        in_degree=None,
        out_degree=None,
        attn_bias=None,
        spatial_pos=None,
        edge_input=None,
        attn_edge_type=None,
    ):
        r"""Computes graph and node embeddings without applying the prediction head."""
        if batched_data is not None:
            x = batched_data["x"]
            in_degree = batched_data["in_degree"]
            out_degree = batched_data["out_degree"]
            attn_bias = batched_data["attn_bias"]
            spatial_pos = batched_data["spatial_pos"]
            edge_input = batched_data.get("edge_input", None)
            attn_edge_type = batched_data.get("attn_edge_type", None)

        h, graph_rep = self.graph_encoder(
            x=x,
            in_degree=in_degree,
            out_degree=out_degree,
            attn_bias=attn_bias,
            spatial_pos=spatial_pos,
            edge_input=edge_input,
            attn_edge_type=attn_edge_type,
            training=False,
        )
        return graph_rep

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name: str = "pcqm4mv1_graphormer_base",
        folder: str = "checkpoints",
        download: bool = True,
        **kwargs,
    ) -> "Graphormer":
        r"""Instantiates a Graphormer model with pre-trained weights."""
        cfg = get_graphormer_config(pretrained_name)
        cfg.update(kwargs)
        model = cls(**cfg)
        load_graphormer_weights(model, pretrained_name=pretrained_name, folder=folder, download=download)
        return model

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"embedding_dim={self.embedding_dim}, "
            f"num_encoder_layers={self.num_encoder_layers}, "
            f"num_attention_heads={self.num_attention_heads}, "
            f"num_classes={self.num_classes})"
        )


# =========================================================================
# Configuration presets & Pre-trained URLs
# =========================================================================

PRETRAINED_MODEL_URLS = {
    "pcqm4mv1_graphormer_base": "https://huggingface.co/clefourrier/graphormer-base-pcqm4mv1/resolve/main/pytorch_model.bin",
    "pcqm4mv2_graphormer_base": "https://huggingface.co/clefourrier/graphormer-base-pcqm4mv2/resolve/main/pytorch_model.bin",
    "pcqm4mv1_graphormer_base_for_molhiv": "https://ml2md.blob.core.windows.net/graphormer-ckpts/checkpoint_base_preln_pcqm4mv1_for_hiv.pt",
}

LEGACY_URLS = {
    "pcqm4mv1_graphormer_base": "https://ml2md.blob.core.windows.net/graphormer-ckpts/checkpoint_best_pcqm4mv1.pt",
    "pcqm4mv2_graphormer_base": "https://ml2md.blob.core.windows.net/graphormer-ckpts/checkpoint_best_pcqm4mv2.pt",
}


def get_graphormer_config(name_or_variant: str) -> Dict[str, Any]:
    r"""Returns configuration dictionary for known Graphormer architectures."""
    key = name_or_variant.lower().replace("-", "_")
    if "slim" in key:
        return {
            "num_encoder_layers": 12,
            "num_attention_heads": 8,
            "embedding_dim": 80,
            "ffn_embedding_dim": 80,
            "pre_layernorm": False,
            "encoder_normalize_before": True,
            "dropout": 0.0,
            "attention_dropout": 0.1,
            "activation_dropout": 0.1,
        }
    elif "large" in key:
        return {
            "num_encoder_layers": 24,
            "num_attention_heads": 32,
            "embedding_dim": 1024,
            "ffn_embedding_dim": 1024,
            "pre_layernorm": False,
            "encoder_normalize_before": True,
            "dropout": 0.0,
            "attention_dropout": 0.1,
            "activation_dropout": 0.1,
        }
    elif "base" in key or "pcqm4m" in key:
        pre_ln = "molhiv" in key or "preln" in key
        num_classes = 1
        if "molhiv" in key:
            num_classes = 2
        return {
            "num_encoder_layers": 12,
            "num_attention_heads": 32,
            "embedding_dim": 768,
            "ffn_embedding_dim": 768,
            "pre_layernorm": pre_ln,
            "encoder_normalize_before": True,
            "dropout": 0.0,
            "attention_dropout": 0.1,
            "activation_dropout": 0.1,
            "num_classes": num_classes,
        }
    else:
        # Standard default architecture
        return {
            "num_encoder_layers": 6,
            "num_attention_heads": 8,
            "embedding_dim": 1024,
            "ffn_embedding_dim": 4096,
            "pre_layernorm": False,
            "encoder_normalize_before": True,
            "dropout": 0.1,
            "attention_dropout": 0.1,
            "activation_dropout": 0.0,
            "num_classes": 1,
        }


def download_graphormer_checkpoint(
    name: str,
    folder: str = "checkpoints",
    log: bool = True,
) -> str:
    r"""Downloads a pre-trained Graphormer checkpoint to the specified folder.

    Args:
        name (str): Pretrained checkpoint name (e.g., ``"pcqm4mv1_graphormer_base"``,
            ``"pcqm4mv2_graphormer_base"``).
        folder (str, optional): Destination folder. (default: ``"checkpoints"``)
        log (bool, optional): Whether to print download progress. (default: ``True``)

    Returns:
        str: Absolute path to the downloaded file.
    """
    from k3_node.data.download import download_url

    clean_name = name.lower().replace("-", "_")
    if clean_name not in PRETRAINED_MODEL_URLS and name not in PRETRAINED_MODEL_URLS:
        raise ValueError(
            f"Unknown pretrained model name '{name}'. Available: {list(PRETRAINED_MODEL_URLS.keys())}"
        )

    matched_key = clean_name if clean_name in PRETRAINED_MODEL_URLS else name
    url = PRETRAINED_MODEL_URLS[matched_key]
    filename = f"{matched_key}.pt"
    local_path = os.path.join(folder, filename)

    if os.path.exists(local_path):
        return local_path

    # Check local repo path if present
    repo_alt = os.path.join("Graphormer-main", "checkpoints", filename)
    if os.path.exists(repo_alt):
        return repo_alt

    try:
        return download_url(url, folder=folder, filename=filename, log=log)
    except Exception as e:
        if matched_key in LEGACY_URLS:
            fallback_url = LEGACY_URLS[matched_key]
            try:
                return download_url(fallback_url, folder=folder, filename=filename, log=log)
            except Exception:
                pass
        raise RuntimeError(f"Failed to download checkpoint for '{name}' from {url}: {e}")


def load_graphormer_weights(
    model: Graphormer,
    checkpoint_path: Optional[str] = None,
    pretrained_name: Optional[str] = None,
    folder: str = "checkpoints",
    download: bool = True,
) -> Graphormer:
    r"""Loads weights from a PyTorch (.pt or .bin) checkpoint into a Keras Graphormer model.

    Args:
        model (Graphormer): Target model instance.
        checkpoint_path (str, optional): Path to .pt or .bin file.
        pretrained_name (str, optional): Name of pre-trained model to load or download.
        folder (str, optional): Checkpoints directory. (default: ``"checkpoints"``)
        download (bool, optional): Whether to download if missing. (default: ``True``)

    Returns:
        Graphormer: The model with loaded weights.
    """
    path_to_load = checkpoint_path

    if path_to_load is None:
        if pretrained_name is None:
            raise ValueError("Either checkpoint_path or pretrained_name must be specified.")
        candidate = os.path.join(folder, f"{pretrained_name}.pt")
        if os.path.isfile(candidate):
            path_to_load = candidate
        elif download:
            path_to_load = download_graphormer_checkpoint(pretrained_name, folder=folder)
        else:
            raise FileNotFoundError(f"Checkpoint for '{pretrained_name}' not found at '{candidate}'.")
    elif not os.path.isfile(path_to_load) and download and pretrained_name:
        path_to_load = download_graphormer_checkpoint(pretrained_name, folder=folder)

    import torch
    import numpy as np

    state = torch.load(path_to_load, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state_dict = state["model"]
    elif isinstance(state, dict):
        state_dict = state
    else:
        raise ValueError(f"Unexpected checkpoint state format: {type(state)}")

    if not model.built:
        model.build(None)

    def _to_tensor(t):
        if hasattr(t, "detach"):
            t = t.detach()
        if hasattr(t, "numpy"):
            t = t.numpy()
        return ops.convert_to_tensor(np.array(t, dtype=np.float32), dtype="float32")

    # Prefix stripping (fairseq checkpoints often have 'encoder.' or 'graph_encoder.')
    clean_dict = {}
    for k, v in state_dict.items():
        ck = k
        if ck.startswith("encoder."):
            ck = ck[len("encoder.") :]
        clean_dict[ck] = v

    # 1. GraphNodeFeature
    gnf = model.graph_encoder.graph_node_feature
    for param_name, layer in [
        ("atom_encoder.weight", gnf.atom_encoder),
        ("in_degree_encoder.weight", gnf.in_degree_encoder),
        ("out_degree_encoder.weight", gnf.out_degree_encoder),
        ("graph_token.weight", gnf.graph_token),
    ]:
        key = f"graph_encoder.graph_node_feature.{param_name}"
        if key in clean_dict:
            layer.weights[0].assign(_to_tensor(clean_dict[key]))

    # 2. GraphAttnBias
    gab = model.graph_encoder.graph_attn_bias
    for param_name, layer in [
        ("edge_encoder.weight", gab.edge_encoder),
        ("spatial_pos_encoder.weight", gab.spatial_pos_encoder),
        ("graph_token_virtual_distance.weight", gab.graph_token_virtual_distance),
    ]:
        key = f"graph_encoder.graph_attn_bias.{param_name}"
        if key in clean_dict:
            layer.weights[0].assign(_to_tensor(clean_dict[key]))

    if hasattr(gab, "edge_dis_encoder"):
        key = "graph_encoder.graph_attn_bias.edge_dis_encoder.weight"
        if key in clean_dict:
            gab.edge_dis_encoder.weights[0].assign(_to_tensor(clean_dict[key]))

    # 3. emb_layer_norm
    if model.graph_encoder.emb_layer_norm is not None:
        eln = model.graph_encoder.emb_layer_norm
        if "graph_encoder.emb_layer_norm.weight" in clean_dict:
            eln.gamma.assign(_to_tensor(clean_dict["graph_encoder.emb_layer_norm.weight"]))
        if "graph_encoder.emb_layer_norm.bias" in clean_dict:
            eln.beta.assign(_to_tensor(clean_dict["graph_encoder.emb_layer_norm.bias"]))

    # 4. Encoder layers
    for i, enc_layer in enumerate(model.graph_encoder.encoder_layers):
        p = f"graph_encoder.layers.{i}"
        # Attention projections
        for proj_name in ["q_proj", "k_proj", "v_proj", "out_proj"]:
            proj = getattr(enc_layer.self_attn, proj_name)
            w_key = f"{p}.self_attn.{proj_name}.weight"
            b_key = f"{p}.self_attn.{proj_name}.bias"
            if w_key in clean_dict:
                proj.kernel.assign(_to_tensor(clean_dict[w_key].t()))
            if b_key in clean_dict and proj.bias is not None:
                proj.bias.assign(_to_tensor(clean_dict[b_key]))

        # self_attn_layer_norm
        if f"{p}.self_attn_layer_norm.weight" in clean_dict:
            enc_layer.self_attn_layer_norm.gamma.assign(
                _to_tensor(clean_dict[f"{p}.self_attn_layer_norm.weight"])
            )
        if f"{p}.self_attn_layer_norm.bias" in clean_dict:
            enc_layer.self_attn_layer_norm.beta.assign(
                _to_tensor(clean_dict[f"{p}.self_attn_layer_norm.bias"])
            )

        # fc1 & fc2
        if f"{p}.fc1.weight" in clean_dict:
            enc_layer.fc1.kernel.assign(_to_tensor(clean_dict[f"{p}.fc1.weight"].t()))
        if f"{p}.fc1.bias" in clean_dict:
            enc_layer.fc1.bias.assign(_to_tensor(clean_dict[f"{p}.fc1.bias"]))
        if f"{p}.fc2.weight" in clean_dict:
            enc_layer.fc2.kernel.assign(_to_tensor(clean_dict[f"{p}.fc2.weight"].t()))
        if f"{p}.fc2.bias" in clean_dict:
            enc_layer.fc2.bias.assign(_to_tensor(clean_dict[f"{p}.fc2.bias"]))

        # final_layer_norm
        if f"{p}.final_layer_norm.weight" in clean_dict:
            enc_layer.final_layer_norm.gamma.assign(
                _to_tensor(clean_dict[f"{p}.final_layer_norm.weight"])
            )
        if f"{p}.final_layer_norm.bias" in clean_dict:
            enc_layer.final_layer_norm.beta.assign(
                _to_tensor(clean_dict[f"{p}.final_layer_norm.bias"])
            )

    # 5. final_layer_norm of encoder stack (if pre_layernorm)
    if model.graph_encoder.final_layer_norm is not None:
        fln = model.graph_encoder.final_layer_norm
        if "graph_encoder.final_layer_norm.weight" in clean_dict:
            fln.gamma.assign(_to_tensor(clean_dict["graph_encoder.final_layer_norm.weight"]))
        if "graph_encoder.final_layer_norm.bias" in clean_dict:
            fln.beta.assign(_to_tensor(clean_dict["graph_encoder.final_layer_norm.bias"]))

    # 6. Prediction Head
    if "lm_head_transform_weight.weight" in clean_dict:
        model.lm_head_transform_weight.kernel.assign(
            _to_tensor(clean_dict["lm_head_transform_weight.weight"].t())
        )
    if "lm_head_transform_weight.bias" in clean_dict:
        model.lm_head_transform_weight.bias.assign(
            _to_tensor(clean_dict["lm_head_transform_weight.bias"])
        )

    if "layer_norm.weight" in clean_dict:
        model.layer_norm.gamma.assign(_to_tensor(clean_dict["layer_norm.weight"]))
    if "layer_norm.bias" in clean_dict:
        model.layer_norm.beta.assign(_to_tensor(clean_dict["layer_norm.bias"]))

    if "embed_out.weight" in clean_dict and hasattr(model, "embed_out"):
        model.embed_out.kernel.assign(_to_tensor(clean_dict["embed_out.weight"].t()))

    if "lm_output_learned_bias" in clean_dict and hasattr(model, "lm_output_learned_bias"):
        model.lm_output_learned_bias.assign(_to_tensor(clean_dict["lm_output_learned_bias"]))

    return model
