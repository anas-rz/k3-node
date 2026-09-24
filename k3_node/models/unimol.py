import math
import os
from typing import Optional, Union, Tuple, List, Dict, Any, Callable

import numpy as np
import keras
from keras import layers, ops

from k3_node.layers.attention.pair_attention import (
    SelfMultiheadAttentionWithPair,
    TransformerEncoderLayerWithPair,
    _get_activation,
)
from k3_node.data.download import download_url


# ==============================================================================
# Pretrained Weights Registry
# ==============================================================================

UNIMOL_PRETRAINED_URLS: Dict[str, Dict[str, str]] = {
    "mol_pre_no_h": {
        "url": "https://github.com/deepmodeling/Uni-Mol/releases/download/v0.1/mol_pre_no_h_220816.pt",
        "filename": "mol_pre_no_h_220816.pt",
        "dict": "mol.dict.txt",
        "dict_url": "https://huggingface.co/dptech/Uni-Mol-Models/resolve/main/mol.dict.txt",
        "description": "Uni-Mol molecular pretraining model (no hydrogen)",
    },
    "mol_pre_all_h": {
        "url": "https://github.com/deepmodeling/Uni-Mol/releases/download/v0.1/mol_pre_all_h_220816.pt",
        "filename": "mol_pre_all_h_220816.pt",
        "dict": "mol.dict.txt",
        "dict_url": "https://huggingface.co/dptech/Uni-Mol-Models/resolve/main/mol.dict.txt",
        "description": "Uni-Mol molecular pretraining model (all hydrogen)",
    },
    "pocket_pre": {
        "url": "https://github.com/deepmodeling/Uni-Mol/releases/download/v0.1/pocket_pre_220816.pt",
        "filename": "pocket_pre_220816.pt",
        "dict": "poc.dict.txt",
        "dict_url": "https://huggingface.co/dptech/Uni-Mol-Models/resolve/main/poc.dict.txt",
        "description": "Uni-Mol candidate protein pocket pretraining model",
    },
    "mp_all_h": {
        "url": "https://huggingface.co/dptech/Uni-Mol-Models/resolve/main/mp_all_h_230313.pt",
        "filename": "mp_all_h_230313.pt",
        "dict": "mp.dict.txt",
        "dict_url": "https://huggingface.co/dptech/Uni-Mol-Models/resolve/main/mp.dict.txt",
        "description": "Uni-Mol crystal Materials Project pretraining model",
    },
    "oled_pre_no_h": {
        "url": "https://huggingface.co/dptech/Uni-Mol-Models/resolve/main/oled_pre_no_h_230101.pt",
        "filename": "oled_pre_no_h_230101.pt",
        "dict": "oled.dict.txt",
        "dict_url": "https://huggingface.co/dptech/Uni-Mol-Models/resolve/main/oled.dict.txt",
        "description": "Uni-Mol OLED molecule pretraining model",
    },
    "qm9": {
        "url": "https://github.com/deepmodeling/Uni-Mol/releases/download/v0.1/qm9_220908.pt",
        "filename": "qm9_220908.pt",
        "dict": "mol.dict.txt",
        "dict_url": "https://huggingface.co/dptech/Uni-Mol-Models/resolve/main/mol.dict.txt",
        "description": "Uni-Mol conformation generation fine-tuned on QM9",
    },
    "drugs": {
        "url": "https://github.com/deepmodeling/Uni-Mol/releases/download/v0.1/drugs_220908.pt",
        "filename": "drugs_220908.pt",
        "dict": "mol.dict.txt",
        "dict_url": "https://huggingface.co/dptech/Uni-Mol-Models/resolve/main/mol.dict.txt",
        "description": "Uni-Mol conformation generation fine-tuned on GEOM-Drugs",
    },
    "binding_pose": {
        "url": "https://github.com/deepmodeling/Uni-Mol/releases/download/v0.1/binding_pose_220908.pt",
        "filename": "binding_pose_220908.pt",
        "dict": "mol.dict.txt",
        "dict_url": "https://huggingface.co/dptech/Uni-Mol-Models/resolve/main/mol.dict.txt",
        "description": "Uni-Mol protein-ligand binding pose prediction",
    },
}

UNIMOL_ALIASES = {
    "molecule": "mol_pre_no_h",
    "molecule_no_h": "mol_pre_no_h",
    "molecule_all_h": "mol_pre_all_h",
    "protein": "pocket_pre",
    "pocket": "pocket_pre",
    "poc_pre": "pocket_pre",
    "crystal": "mp_all_h",
    "mp": "mp_all_h",
    "oled": "oled_pre_no_h",
}


# ==============================================================================
# Layers
# ==============================================================================

class GaussianLayer(layers.Layer):
    r"""Gaussian basis function (GBF) expansion over pairwise distances modulated by edge types.

    Args:
        num_kernel (int, optional): Number of Gaussian kernels. (default: ``128``)
        edge_types (int, optional): Number of distinct pairwise edge types. (default: ``1024``)
        **kwargs: Additional layer arguments.
    """

    def __init__(self, num_kernel: int = 128, edge_types: int = 1024, **kwargs):
        super().__init__(**kwargs)
        self.num_kernel = num_kernel
        self.edge_types = edge_types

        self.means = layers.Embedding(1, num_kernel, embeddings_initializer="uniform", name="means")
        self.stds = layers.Embedding(1, num_kernel, embeddings_initializer="uniform", name="stds")
        self.mul = layers.Embedding(edge_types, 1, embeddings_initializer="ones", name="mul")
        self.bias = layers.Embedding(edge_types, 1, embeddings_initializer="zeros", name="bias")

    def build(self, input_shape=None):
        if not self.built:
            self.means.build(None)
            self.stds.build(None)
            self.mul.build(None)
            self.bias.build(None)
        super().build(input_shape)

    def call(self, dist, edge_type):
        r"""
        Args:
            dist (Tensor): Pairwise distance matrix of shape ``[batch_size, seq_len, seq_len]``.
            edge_type (Tensor): Pairwise edge type indices of shape ``[batch_size, seq_len, seq_len]``.

        Returns:
            Tensor: Gaussian basis expansion of shape ``[batch_size, seq_len, seq_len, num_kernel]``.
        """
        mul = ops.cast(self.mul(edge_type), dist.dtype)
        bias = ops.cast(self.bias(edge_type), dist.dtype)

        # x: [B, N, N, 1]
        x = mul * ops.expand_dims(dist, axis=-1) + bias

        # Means and stds: [num_kernel]
        zero_idx = ops.zeros((1,), dtype="int32")
        mean = ops.reshape(self.means(zero_idx), (-1,))
        std = ops.abs(ops.reshape(self.stds(zero_idx), (-1,))) + 1e-5

        a = math.sqrt(2.0 * math.pi)
        diff = (x - mean) / std
        return ops.exp(-0.5 * ops.power(diff, 2)) / (a * std)


class NumericalEmbed(layers.Layer):
    r"""Numerical embedding layer for continuous edge features."""

    def __init__(self, num_kernel: int = 128, edge_types: int = 1024, activation_fn: str = "gelu", **kwargs):
        super().__init__(**kwargs)
        self.num_kernel = num_kernel
        self.edge_types = edge_types
        self.mul = layers.Embedding(edge_types, 1, name="mul")
        self.bias = layers.Embedding(edge_types, 1, name="bias")
        self.w_edge = layers.Embedding(edge_types, num_kernel, name="w_edge")
        self.proj = NonLinearHead(1, num_kernel, activation_fn=activation_fn, hidden=2 * num_kernel, name="proj")
        self.ln = layers.LayerNormalization(axis=-1, epsilon=1e-5, name="ln")

    def build(self, input_shape=None):
        if not self.built:
            self.mul.build(None)
            self.bias.build(None)
            self.w_edge.build(None)
            self.proj.build((None, None, None, 1))
            self.ln.build((None, None, None, self.num_kernel))
        super().build(input_shape)

    def call(self, dist, edge_type):
        mul = ops.cast(self.mul(edge_type), dist.dtype)
        bias = ops.cast(self.bias(edge_type), dist.dtype)
        w_edge = ops.cast(self.w_edge(edge_type), dist.dtype)

        edge_feat = mul * ops.expand_dims(dist, axis=-1) + bias
        edge_feat = self.proj(edge_feat)
        edge_feat = edge_feat + w_edge
        return self.ln(edge_feat)


class NonLinearHead(layers.Layer):
    r"""Two-layer feed-forward network with activation for feature projection."""

    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        activation_fn: Union[str, Callable] = "gelu",
        hidden: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.hidden = hidden or input_dim
        self.activation_fn_name = activation_fn

        self.linear1 = layers.Dense(self.hidden, name="linear1")
        self.act = _get_activation(activation_fn)
        self.linear2 = layers.Dense(out_dim, name="linear2")

    def build(self, input_shape=None):
        if not self.built:
            self.linear1.build((None, None, None, self.input_dim) if len(input_shape or ()) == 4 else (None, None, self.input_dim))
            self.linear2.build((None, None, None, self.hidden) if len(input_shape or ()) == 4 else (None, None, self.hidden))
        super().build(input_shape)

    def call(self, x):
        x = self.linear1(x)
        if self.act is not None:
            x = self.act(x)
        x = self.linear2(x)
        return x


class DistanceHead(layers.Layer):
    r"""Symmetrized distance prediction head from pair representations."""

    def __init__(self, heads: int, activation_fn: Union[str, Callable] = "gelu", **kwargs):
        super().__init__(**kwargs)
        self.heads = heads
        self.dense = layers.Dense(heads, name="dense")
        self.act = _get_activation(activation_fn)
        self.layer_norm = layers.LayerNormalization(axis=-1, epsilon=1e-5, name="layer_norm")
        self.out_proj = layers.Dense(1, name="out_proj")

    def build(self, input_shape=None):
        if not self.built:
            self.dense.build((None, None, None, self.heads))
            self.layer_norm.build((None, None, None, self.heads))
            self.out_proj.build((None, None, None, self.heads))
        super().build(input_shape)

    def call(self, x):
        r"""
        Args:
            x (Tensor): Pair tensor of shape ``[batch_size, seq_len, seq_len, heads]``.

        Returns:
            Tensor: Symmetrized predicted distances ``[batch_size, seq_len, seq_len]``.
        """
        x = self.dense(x)
        if self.act is not None:
            x = self.act(x)
        x = self.layer_norm(x)
        x = ops.squeeze(self.out_proj(x), axis=-1)  # [B, N, N]
        return 0.5 * (x + ops.transpose(x, (0, 2, 1)))


class LinearHead(layers.Layer):
    r"""Linear classification/regression head."""

    def __init__(self, input_dim: int, num_classes: int, pooler_dropout: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.pooler_dropout = pooler_dropout

        self.dropout = layers.Dropout(pooler_dropout) if pooler_dropout > 0.0 else None
        self.out_proj = layers.Dense(num_classes, name="out_proj")

    def build(self, input_shape=None):
        if not self.built:
            self.out_proj.build((None, self.input_dim))
        super().build(input_shape)

    def call(self, features, training: bool = False):
        x = features
        if self.dropout is not None:
            x = self.dropout(x, training=training)
        return self.out_proj(x)


class ClassificationHead(layers.Layer):
    r"""Two-layer sentence/graph-level classification head."""

    def __init__(
        self,
        input_dim: int,
        inner_dim: int,
        num_classes: int,
        activation_fn: Union[str, Callable] = "gelu",
        pooler_dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.inner_dim = inner_dim
        self.num_classes = num_classes
        self.pooler_dropout = pooler_dropout

        self.dense = layers.Dense(inner_dim, name="dense")
        self.act = _get_activation(activation_fn)
        self.dropout = layers.Dropout(pooler_dropout) if pooler_dropout > 0.0 else None
        self.out_proj = layers.Dense(num_classes, name="out_proj")

    def build(self, input_shape=None):
        if not self.built:
            self.dense.build((None, self.input_dim))
            self.out_proj.build((None, self.inner_dim))
        super().build(input_shape)

    def call(self, features, training: bool = False):
        x = features
        if len(ops.shape(features)) == 3:
            x = features[:, 0, :]  # CLS token
        if self.dropout is not None:
            x = self.dropout(x, training=training)
        x = self.dense(x)
        if self.act is not None:
            x = self.act(x)
        if self.dropout is not None:
            x = self.dropout(x, training=training)
        return self.out_proj(x)


class MaskLMHead(layers.Layer):
    r"""Masked language modeling head for predicting masked atom tokens."""

    def __init__(
        self,
        embed_dim: int,
        output_dim: int,
        activation_fn: Union[str, Callable] = "gelu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.dense = layers.Dense(embed_dim, name="dense")
        self.act = _get_activation(activation_fn)
        self.layer_norm = layers.LayerNormalization(axis=-1, epsilon=1e-5, name="layer_norm")
        self.out_proj = layers.Dense(output_dim, name="out_proj")

    def build(self, input_shape=None):
        if not self.built:
            self.dense.build((None, None, self.embed_dim))
            self.layer_norm.build((None, None, self.embed_dim))
            self.out_proj.build((None, None, self.embed_dim))
        super().build(input_shape)

    def call(self, features, masked_tokens=None):
        x = self.dense(features)
        if self.act is not None:
            x = self.act(x)
        x = self.layer_norm(x)
        x = self.out_proj(x)
        return x


# ==============================================================================
# Backbone Transformer Encoder
# ==============================================================================

class UniMolTransformerEncoder(layers.Layer):
    r"""Transformer Encoder backbone for Uni-Mol with pair attention bias propagation."""

    def __init__(
        self,
        encoder_layers: int = 15,
        embed_dim: int = 512,
        ffn_embed_dim: int = 2048,
        attention_heads: int = 64,
        emb_dropout: float = 0.1,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        max_seq_len: int = 512,
        activation_fn: Union[str, Callable] = "gelu",
        post_ln: bool = False,
        no_final_head_layer_norm: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_layers = encoder_layers
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.attention_heads = attention_heads
        self.emb_dropout_rate = emb_dropout
        self.post_ln = post_ln

        self.emb_layer_norm = layers.LayerNormalization(axis=-1, epsilon=1e-5, name="emb_layer_norm")
        self.emb_dropout = layers.Dropout(emb_dropout) if emb_dropout > 0.0 else None

        self.final_layer_norm = None if post_ln else layers.LayerNormalization(axis=-1, epsilon=1e-5, name="final_layer_norm")
        self.final_head_layer_norm = None if no_final_head_layer_norm else layers.LayerNormalization(axis=-1, epsilon=1e-5, name="final_head_layer_norm")

        self.layers_list = [
            TransformerEncoderLayerWithPair(
                embed_dim=embed_dim,
                ffn_embed_dim=ffn_embed_dim,
                attention_heads=attention_heads,
                dropout=dropout,
                attention_dropout=attention_dropout,
                activation_dropout=activation_dropout,
                activation_fn=activation_fn,
                post_ln=post_ln,
                name=f"layer_{i}",
            )
            for i in range(encoder_layers)
        ]

    def build(self, input_shape=None):
        if not self.built:
            self.emb_layer_norm.build((None, None, self.embed_dim))
            if self.final_layer_norm is not None:
                self.final_layer_norm.build((None, None, self.embed_dim))
            if self.final_head_layer_norm is not None:
                self.final_head_layer_norm.build((None, None, None, self.attention_heads))
            for layer in self.layers_list:
                layer.build((None, None, self.embed_dim))
        super().build(input_shape)

    def call(self, emb, attn_mask=None, padding_mask=None, training: bool = False):
        shape = ops.shape(emb)
        bsz = shape[0]
        seq_len = shape[1]

        x = self.emb_layer_norm(emb)
        if self.emb_dropout is not None:
            x = self.emb_dropout(x, training=training)

        if padding_mask is not None:
            mask_expanded = ops.expand_dims(ops.cast(padding_mask, x.dtype), axis=-1)
            x = x * (1.0 - mask_expanded)

        if attn_mask is not None:
            mask_shape = ops.shape(attn_mask)
            if len(mask_shape) == 4 and mask_shape[-1] == self.attention_heads:
                attn_mask = ops.transpose(attn_mask, (0, 3, 1, 2))
            elif len(mask_shape) == 3:
                attn_mask = ops.reshape(attn_mask, (bsz, self.attention_heads, seq_len, seq_len))

        input_attn_mask = attn_mask
        curr_attn_mask = attn_mask

        for enc_layer in self.layers_list:
            x, curr_attn_mask, _ = enc_layer(
                x,
                attn_bias=curr_attn_mask,
                padding_mask=padding_mask,
                return_attn=True,
                training=training,
            )

        if self.final_layer_norm is not None:
            x = self.final_layer_norm(x)

        # Delta pair representation
        delta_pair_repr = curr_attn_mask - input_attn_mask

        # Reshape to [bsz, seq_len, seq_len, attention_heads]
        pair_shape = ops.shape(curr_attn_mask)
        if len(pair_shape) == 3:
            curr_attn_mask = ops.reshape(curr_attn_mask, (bsz, self.attention_heads, seq_len, seq_len))
            delta_pair_repr = ops.reshape(delta_pair_repr, (bsz, self.attention_heads, seq_len, seq_len))

        curr_attn_mask = ops.transpose(curr_attn_mask, (0, 2, 3, 1))
        delta_pair_repr = ops.transpose(delta_pair_repr, (0, 2, 3, 1))

        if self.final_head_layer_norm is not None:
            delta_pair_repr = self.final_head_layer_norm(delta_pair_repr)

        return x, curr_attn_mask, delta_pair_repr


# ==============================================================================
# Uni-Mol Model
# ==============================================================================

class UniMolModel(keras.Model):
    r"""Multi-backend Uni-Mol model for 3D molecular representation learning and property prediction.

    Supports molecular pretraining, candidate pocket pretraining, crystal, and OLED configurations.

    Args:
        output_dim (int, optional): Number of task output dimensions / classes. (default: ``2``)
        data_type (str, optional): Data domain (``"molecule"``, ``"protein"``, ``"crystal"``, ``"oled"``). (default: ``"molecule"``)
        vocab_size (int, optional): Vocabulary size for token dictionary. (default: ``512``)
        encoder_layers (int, optional): Number of transformer encoder layers. (default: ``15``)
        encoder_embed_dim (int, optional): Node embedding dimension. (default: ``512``)
        encoder_ffn_embed_dim (int, optional): FFN hidden dimension. (default: ``2048``)
        encoder_attention_heads (int, optional): Number of attention heads. (default: ``64``)
        kernel (str, optional): GBF kernel type (``"gaussian"`` or ``"numerical"``). (default: ``"gaussian"``)
        num_kernel (int, optional): Number of radial kernels. (default: ``128``)
        pooler_dropout (float, optional): Dropout for classification head. (default: ``0.0``)
        activation_fn (str, optional): Activation function name. (default: ``"gelu"``)
        post_ln (bool, optional): Post-LN flag. (default: ``False``)
        **kwargs: Additional model arguments.
    """

    def __init__(
        self,
        output_dim: int = 2,
        data_type: str = "molecule",
        vocab_size: int = 512,
        encoder_layers: int = 15,
        encoder_embed_dim: int = 512,
        encoder_ffn_embed_dim: int = 2048,
        encoder_attention_heads: int = 64,
        kernel: str = "gaussian",
        num_kernel: int = 128,
        pooler_dropout: float = 0.0,
        activation_fn: str = "gelu",
        post_ln: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.data_type = data_type
        self.vocab_size = vocab_size
        self.encoder_layers = encoder_layers
        self.encoder_embed_dim = encoder_embed_dim
        self.encoder_ffn_embed_dim = encoder_ffn_embed_dim
        self.encoder_attention_heads = encoder_attention_heads
        self.kernel_type = kernel
        self.num_kernel = num_kernel
        self.pooler_dropout = pooler_dropout
        self.activation_fn_name = activation_fn
        self.post_ln = post_ln

        self.padding_idx = 0

        self.embed_tokens = layers.Embedding(vocab_size, encoder_embed_dim, name="embed_tokens")

        n_edge_type = 1024
        if kernel == "gaussian":
            self.gbf = GaussianLayer(num_kernel, n_edge_type, name="gbf")
        else:
            self.gbf = NumericalEmbed(num_kernel, n_edge_type, activation_fn=activation_fn, name="gbf")

        self.gbf_proj = NonLinearHead(
            num_kernel, encoder_attention_heads, activation_fn=activation_fn, name="gbf_proj"
        )

        self.encoder = UniMolTransformerEncoder(
            encoder_layers=encoder_layers,
            embed_dim=encoder_embed_dim,
            ffn_embed_dim=encoder_ffn_embed_dim,
            attention_heads=encoder_attention_heads,
            activation_fn=activation_fn,
            post_ln=post_ln,
            name="encoder",
        )

        self.classification_head = LinearHead(
            input_dim=encoder_embed_dim,
            num_classes=output_dim,
            pooler_dropout=pooler_dropout,
            name="classification_head",
        )

        self.pair2coord_proj = NonLinearHead(
            encoder_attention_heads, 1, activation_fn=activation_fn, name="pair2coord_proj"
        )
        self.dist_head = DistanceHead(
            encoder_attention_heads, activation_fn=activation_fn, name="dist_head"
        )
        self.lm_head = MaskLMHead(
            encoder_embed_dim, vocab_size, activation_fn=activation_fn, name="lm_head"
        )

    def build(self, input_shape=None):
        if not self.built:
            self.embed_tokens.build(None)
            self.gbf.build(None)
            self.gbf_proj.build((None, None, None, self.num_kernel))
            self.encoder.build(None)
            self.classification_head.build((None, self.encoder_embed_dim))
            self.pair2coord_proj.build((None, None, None, self.encoder_attention_heads))
            self.dist_head.build((None, None, None, self.encoder_attention_heads))
            self.lm_head.build((None, None, self.encoder_embed_dim))
        super().build(input_shape)

    def call(
        self,
        src_tokens,
        src_distance=None,
        src_coord=None,
        src_edge_type=None,
        padding_mask=None,
        return_repr: bool = False,
        return_atomic_reprs: bool = False,
        features_only: bool = False,
        training: bool = False,
    ):
        if isinstance(src_tokens, dict):
            src_distance = src_tokens.get("src_distance", src_distance)
            src_coord = src_tokens.get("src_coord", src_coord)
            src_edge_type = src_tokens.get("src_edge_type", src_edge_type)
            padding_mask = src_tokens.get("padding_mask", padding_mask)
            src_tokens = src_tokens.get("src_tokens", src_tokens.get("tokens"))
        elif isinstance(src_tokens, (tuple, list)):
            if len(src_tokens) == 2:
                src_tokens, src_coord = src_tokens
            elif len(src_tokens) >= 3:
                src_tokens, src_distance, src_coord = src_tokens[:3]
        r"""Forward pass for Uni-Mol.

        Args:
            src_tokens (Tensor): Token indices of shape ``[batch_size, seq_len]``.
            src_distance (Tensor, optional): Pairwise distances ``[batch_size, seq_len, seq_len]``.
            src_coord (Tensor, optional): 3D coordinates ``[batch_size, seq_len, 3]``.
            src_edge_type (Tensor, optional): Pairwise edge types ``[batch_size, seq_len, seq_len]``.
            padding_mask (Tensor, optional): Padding boolean mask ``[batch_size, seq_len]``.
            return_repr (bool, optional): Return CLS/atomic representations.
            features_only (bool, optional): Only return representations, skipping heads.
            training (bool, optional): Training mode flag.
        """
        shape = ops.shape(src_tokens)
        bsz = shape[0]
        seq_len = shape[1]

        # Auto-compute padding_mask if not provided
        if padding_mask is None:
            padding_mask = ops.equal(src_tokens, self.padding_idx)

        # Auto-compute src_distance from src_coord if distance is missing
        if src_distance is None and src_coord is not None:
            diff = ops.expand_dims(src_coord, axis=2) - ops.expand_dims(src_coord, axis=1)
            src_distance = ops.sqrt(ops.sum(ops.power(diff, 2), axis=-1) + 1e-10)
        elif src_distance is None:
            src_distance = ops.zeros((bsz, seq_len, seq_len), dtype="float32")

        # Auto-compute src_edge_type from tokens if missing
        if src_edge_type is None:
            n_types = 32
            src_edge_type = ops.expand_dims(src_tokens, axis=-1) * n_types + ops.expand_dims(src_tokens, axis=1)
            src_edge_type = ops.cast(ops.mod(src_edge_type, 1024), "int32")

        # 1. Embeddings & GBF
        x = self.embed_tokens(src_tokens)
        gbf_feature = self.gbf(src_distance, src_edge_type)
        graph_attn_bias = self.gbf_proj(gbf_feature)  # [B, N, N, H]

        # 2. Encoder
        encoder_rep, encoder_pair_rep, delta_pair_rep = self.encoder(
            x,
            attn_mask=graph_attn_bias,
            padding_mask=padding_mask,
            training=training,
        )

        cls_repr = encoder_rep[:, 0, :]

        if return_repr:
            res = {"cls_repr": cls_repr, "encoder_rep": encoder_rep, "pair_rep": encoder_pair_rep}
            return res

        if features_only:
            return encoder_rep, encoder_pair_rep

        # Classification / Regression logits
        logits = self.classification_head(cls_repr, training=training)
        return logits


class UniMolConfGenModel(UniMolModel):
    r"""Uni-Mol Conformation Generation Model for iterative 3D geometry prediction."""

    def call(
        self,
        src_tokens,
        src_distance=None,
        src_coord=None,
        src_edge_type=None,
        padding_mask=None,
        training: bool = False,
    ):
        if isinstance(src_tokens, dict):
            src_distance = src_tokens.get("src_distance", src_distance)
            src_coord = src_tokens.get("src_coord", src_coord)
            src_edge_type = src_tokens.get("src_edge_type", src_edge_type)
            padding_mask = src_tokens.get("padding_mask", padding_mask)
            src_tokens = src_tokens.get("src_tokens", src_tokens.get("tokens"))
        elif isinstance(src_tokens, (tuple, list)):
            if len(src_tokens) == 2:
                src_tokens, src_coord = src_tokens
            elif len(src_tokens) >= 3:
                src_tokens, src_distance, src_coord = src_tokens[:3]

        if padding_mask is None:
            padding_mask = ops.equal(src_tokens, self.padding_idx)

        shape = ops.shape(src_tokens)
        bsz = shape[0]
        seq_len = shape[1]

        if src_coord is None:
            src_coord = ops.zeros((bsz, seq_len, 3), dtype="float32")

        if src_distance is None:
            diff = ops.expand_dims(src_coord, axis=2) - ops.expand_dims(src_coord, axis=1)
            src_distance = ops.sqrt(ops.sum(ops.power(diff, 2), axis=-1) + 1e-10)

        if src_edge_type is None:
            src_edge_type = ops.cast(ops.mod(ops.expand_dims(src_tokens, axis=-1) * 32 + ops.expand_dims(src_tokens, axis=1), 1024), "int32")

        x = self.embed_tokens(src_tokens)
        gbf_feature = self.gbf(src_distance, src_edge_type)
        graph_attn_bias = self.gbf_proj(gbf_feature)

        encoder_rep, encoder_pair_rep, delta_pair_rep = self.encoder(
            x,
            attn_mask=graph_attn_bias,
            padding_mask=padding_mask,
            training=training,
        )

        # Coordinate update from delta_pair_rep
        attn_probs = self.pair2coord_proj(delta_pair_rep)  # [B, N, N, 1]
        delta_pos = ops.expand_dims(src_coord, axis=1) - ops.expand_dims(src_coord, axis=2)
        coord_update = delta_pos * attn_probs
        updated_coord = src_coord + ops.sum(coord_update, axis=2)

        pred_dist = self.dist_head(encoder_pair_rep)
        return updated_coord, pred_dist


class UniMolDockingModel(UniMolModel):
    r"""Uni-Mol Protein-Ligand Binding Pose Prediction Model."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Cross-layer coordinate prediction head
        self.docking_coord_head = NonLinearHead(
            self.encoder_attention_heads, 1, activation_fn="gelu", name="docking_coord_head"
        )

    def build(self, input_shape=None):
        if not self.built:
            self.docking_coord_head.build((None, None, None, self.encoder_attention_heads))
        super().build(input_shape)

    def call(
        self,
        src_tokens,
        src_distance=None,
        src_coord=None,
        src_edge_type=None,
        padding_mask=None,
        training: bool = False,
    ):
        if isinstance(src_tokens, dict):
            src_distance = src_tokens.get("src_distance", src_distance)
            src_coord = src_tokens.get("src_coord", src_coord)
            src_edge_type = src_tokens.get("src_edge_type", src_edge_type)
            padding_mask = src_tokens.get("padding_mask", padding_mask)
            src_tokens = src_tokens.get("src_tokens", src_tokens.get("tokens"))
        elif isinstance(src_tokens, (tuple, list)):
            if len(src_tokens) == 2:
                src_tokens, src_coord = src_tokens
            elif len(src_tokens) >= 3:
                src_tokens, src_distance, src_coord = src_tokens[:3]

        if padding_mask is None:
            padding_mask = ops.equal(src_tokens, self.padding_idx)

        shape = ops.shape(src_tokens)
        bsz = shape[0]
        seq_len = shape[1]

        if src_coord is None:
            src_coord = ops.zeros((bsz, seq_len, 3), dtype="float32")

        if src_distance is None:
            diff = ops.expand_dims(src_coord, axis=2) - ops.expand_dims(src_coord, axis=1)
            src_distance = ops.sqrt(ops.sum(ops.power(diff, 2), axis=-1) + 1e-10)

        if src_edge_type is None:
            src_edge_type = ops.cast(ops.mod(ops.expand_dims(src_tokens, axis=-1) * 32 + ops.expand_dims(src_tokens, axis=1), 1024), "int32")

        x = self.embed_tokens(src_tokens)
        gbf_feature = self.gbf(src_distance, src_edge_type)
        graph_attn_bias = self.gbf_proj(gbf_feature)

        encoder_rep, encoder_pair_rep, delta_pair_rep = self.encoder(
            x,
            attn_mask=graph_attn_bias,
            padding_mask=padding_mask,
            training=training,
        )

        probs = self.docking_coord_head(delta_pair_rep)
        diff_coord = ops.expand_dims(src_coord, axis=1) - ops.expand_dims(src_coord, axis=2)
        pose_update = src_coord + ops.sum(diff_coord * probs, axis=2)
        pred_dist = self.dist_head(encoder_pair_rep)
        return pose_update, pred_dist


# ==============================================================================
# Helper Functions: Download & Load Weights
# ==============================================================================

def download_unimol_checkpoint(
    name: str = "mol_pre_no_h",
    folder: str = "checkpoints",
    log: bool = True,
) -> str:
    r"""Downloads a pre-trained Uni-Mol checkpoint (.pt).

    Args:
        name (str): Pretrained checkpoint name or alias (e.g. ``"mol_pre_no_h"``,
            ``"mol_pre_all_h"``, ``"pocket_pre"``, ``"mp_all_h"``, ``"oled_pre_no_h"``,
            ``"qm9"``, ``"drugs"``, ``"binding_pose"``).
        folder (str, optional): Target directory to save the checkpoint. (default: ``"checkpoints"``)
        log (bool, optional): Whether to print download progress. (default: ``True``)

    Returns:
        str: Absolute path to the downloaded checkpoint file.
    """
    clean_name = name.strip().lower()
    if clean_name in UNIMOL_ALIASES:
        clean_name = UNIMOL_ALIASES[clean_name]

    if clean_name not in UNIMOL_PRETRAINED_URLS:
        for k, v in UNIMOL_PRETRAINED_URLS.items():
            if v["filename"] == name or k.lower() == clean_name:
                clean_name = k
                break

    if clean_name not in UNIMOL_PRETRAINED_URLS:
        raise ValueError(
            f"Unknown Uni-Mol checkpoint '{name}'. Available: {list(UNIMOL_PRETRAINED_URLS.keys())}"
        )

    info = UNIMOL_PRETRAINED_URLS[clean_name]
    filename = info["filename"]
    local_path = os.path.join(folder, filename)

    if os.path.exists(local_path):
        return local_path

    # Check alternative local paths
    alt_paths = [
        os.path.join("Uni-Mol", "unimol", filename),
        os.path.join("Uni-Mol", "unimol_tools", "unimol_tools", "weights", filename),
    ]
    for alt in alt_paths:
        if os.path.exists(alt):
            return alt

    return download_url(info["url"], folder=folder, filename=filename, log=log)


def load_unimol_weights(
    model: UniMolModel,
    checkpoint_path: Optional[str] = None,
    pretrained_name: Optional[str] = None,
    folder: str = "checkpoints",
    download: bool = True,
) -> UniMolModel:
    r"""Loads pre-trained weights from a PyTorch checkpoint (.pt) into a Keras 3 UniMolModel.

    Args:
        model (UniMolModel): The target UniMolModel instance.
        checkpoint_path (str, optional): Local path to .pt file or checkpoint name.
        pretrained_name (str, optional): Pretrained model identifier.
        folder (str, optional): Directory to store downloaded checkpoints. (default: ``"checkpoints"``)
        download (bool, optional): Whether to download checkpoint if missing locally. (default: ``True``)

    Returns:
        UniMolModel: The model with loaded weights.
    """
    path_to_load = checkpoint_path

    candidate = pretrained_name or checkpoint_path
    if candidate:
        clean = candidate.strip().lower()
        if clean in UNIMOL_ALIASES:
            candidate = UNIMOL_ALIASES[clean]

    if path_to_load is None:
        if candidate is None:
            raise ValueError("Either checkpoint_path or pretrained_name must be specified.")
        if candidate in UNIMOL_PRETRAINED_URLS:
            fname = UNIMOL_PRETRAINED_URLS[candidate]["filename"]
            local_target = os.path.join(folder, fname)
            if os.path.isfile(local_target):
                path_to_load = local_target
            elif download:
                path_to_load = download_unimol_checkpoint(candidate, folder=folder)
            else:
                raise FileNotFoundError(f"Checkpoint for '{candidate}' not found at '{local_target}'.")
        else:
            raise ValueError(f"Unknown checkpoint '{candidate}'.")
    elif not os.path.isfile(path_to_load) and download and candidate in UNIMOL_PRETRAINED_URLS:
        path_to_load = download_unimol_checkpoint(candidate, folder=folder)

    import torch

    state = torch.load(path_to_load, map_location="cpu")
    if isinstance(state, dict):
        if "model" in state:
            state_dict = state["model"]
        elif "model_state_dict" in state:
            state_dict = state["model_state_dict"]
        else:
            state_dict = state
    else:
        raise ValueError(f"Expected dict/state_dict in checkpoint, got {type(state)}")

    if not model.built:
        model.build(None)

    def _to_tensor(t):
        if hasattr(t, "detach"):
            t = t.detach()
        if hasattr(t, "cpu"):
            t = t.cpu()
        if hasattr(t, "numpy"):
            t = t.numpy()
        return ops.convert_to_tensor(np.array(t, dtype=np.float32), dtype="float32")

    # 1. Embeddings
    if "embed_tokens.weight" in state_dict:
        w = state_dict["embed_tokens.weight"]
        model.embed_tokens.embeddings.assign(_to_tensor(w))

    # 2. GBF
    if "gbf.means.weight" in state_dict:
        model.gbf.means.embeddings.assign(_to_tensor(state_dict["gbf.means.weight"]))
    if "gbf.stds.weight" in state_dict:
        model.gbf.stds.embeddings.assign(_to_tensor(state_dict["gbf.stds.weight"]))
    if "gbf.mul.weight" in state_dict:
        model.gbf.mul.embeddings.assign(_to_tensor(state_dict["gbf.mul.weight"]))
    if "gbf.bias.weight" in state_dict:
        model.gbf.bias.embeddings.assign(_to_tensor(state_dict["gbf.bias.weight"]))

    # 3. GBF Proj
    if "gbf_proj.linear1.weight" in state_dict:
        model.gbf_proj.linear1.kernel.assign(_to_tensor(state_dict["gbf_proj.linear1.weight"].t()))
    if "gbf_proj.linear1.bias" in state_dict:
        model.gbf_proj.linear1.bias.assign(_to_tensor(state_dict["gbf_proj.linear1.bias"]))
    if "gbf_proj.linear2.weight" in state_dict:
        model.gbf_proj.linear2.kernel.assign(_to_tensor(state_dict["gbf_proj.linear2.weight"].t()))
    if "gbf_proj.linear2.bias" in state_dict:
        model.gbf_proj.linear2.bias.assign(_to_tensor(state_dict["gbf_proj.linear2.bias"]))

    # 4. Encoder Layers
    for i, enc_layer in enumerate(model.encoder.layers_list):
        prefix = f"encoder.layers.{i}"
        alt_prefix = f"layers.{i}"

        def _get(name):
            if f"{prefix}.{name}" in state_dict:
                return state_dict[f"{prefix}.{name}"]
            elif f"{alt_prefix}.{name}" in state_dict:
                return state_dict[f"{alt_prefix}.{name}"]
            return None

        # Self-Attention
        in_w = _get("self_attn.in_proj.weight")
        if in_w is not None:
            enc_layer.self_attn.in_proj.kernel.assign(_to_tensor(in_w.t()))
        in_b = _get("self_attn.in_proj.bias")
        if in_b is not None:
            enc_layer.self_attn.in_proj.bias.assign(_to_tensor(in_b))

        out_w = _get("self_attn.out_proj.weight")
        if out_w is not None:
            enc_layer.self_attn.out_proj.kernel.assign(_to_tensor(out_w.t()))
        out_b = _get("self_attn.out_proj.bias")
        if out_b is not None:
            enc_layer.self_attn.out_proj.bias.assign(_to_tensor(out_b))

        # Layer norms
        attn_ln_w = _get("self_attn_layer_norm.weight")
        if attn_ln_w is not None and enc_layer.self_attn_layer_norm.gamma is not None:
            enc_layer.self_attn_layer_norm.gamma.assign(_to_tensor(attn_ln_w))
        attn_ln_b = _get("self_attn_layer_norm.bias")
        if attn_ln_b is not None and enc_layer.self_attn_layer_norm.beta is not None:
            enc_layer.self_attn_layer_norm.beta.assign(_to_tensor(attn_ln_b))

        # FFN
        fc1_w = _get("fc1.weight")
        if fc1_w is not None:
            enc_layer.fc1.kernel.assign(_to_tensor(fc1_w.t()))
        fc1_b = _get("fc1.bias")
        if fc1_b is not None:
            enc_layer.fc1.bias.assign(_to_tensor(fc1_b))

        fc2_w = _get("fc2.weight")
        if fc2_w is not None:
            enc_layer.fc2.kernel.assign(_to_tensor(fc2_w.t()))
        fc2_b = _get("fc2.bias")
        if fc2_b is not None:
            enc_layer.fc2.bias.assign(_to_tensor(fc2_b))

        final_ln_w = _get("final_layer_norm.weight")
        if final_ln_w is not None and enc_layer.final_layer_norm.gamma is not None:
            enc_layer.final_layer_norm.gamma.assign(_to_tensor(final_ln_w))
        final_ln_b = _get("final_layer_norm.bias")
        if final_ln_b is not None and enc_layer.final_layer_norm.beta is not None:
            enc_layer.final_layer_norm.beta.assign(_to_tensor(final_ln_b))

    # Encoder global norms
    for key, attr in [
        ("encoder.emb_layer_norm.weight", model.encoder.emb_layer_norm.gamma),
        ("encoder.emb_layer_norm.bias", model.encoder.emb_layer_norm.beta),
    ]:
        if key in state_dict and attr is not None:
            attr.assign(_to_tensor(state_dict[key]))

    if model.encoder.final_layer_norm is not None:
        if "encoder.final_layer_norm.weight" in state_dict:
            model.encoder.final_layer_norm.gamma.assign(_to_tensor(state_dict["encoder.final_layer_norm.weight"]))
        if "encoder.final_layer_norm.bias" in state_dict:
            model.encoder.final_layer_norm.beta.assign(_to_tensor(state_dict["encoder.final_layer_norm.bias"]))

    if model.encoder.final_head_layer_norm is not None:
        if "encoder.final_head_layer_norm.weight" in state_dict:
            model.encoder.final_head_layer_norm.gamma.assign(_to_tensor(state_dict["encoder.final_head_layer_norm.weight"]))
        if "encoder.final_head_layer_norm.bias" in state_dict:
            model.encoder.final_head_layer_norm.beta.assign(_to_tensor(state_dict["encoder.final_head_layer_norm.bias"]))

    # Classification / linear head
    if hasattr(model, "classification_head"):
        for k in ["classification_head.out_proj.weight", "classification_heads.target.out_proj.weight"]:
            if k in state_dict and hasattr(model.classification_head, "out_proj"):
                model.classification_head.out_proj.kernel.assign(_to_tensor(state_dict[k].t()))
        for k in ["classification_head.out_proj.bias", "classification_heads.target.out_proj.bias"]:
            if k in state_dict and hasattr(model.classification_head, "out_proj"):
                model.classification_head.out_proj.bias.assign(_to_tensor(state_dict[k]))

    return model
