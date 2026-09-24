import math
import os
from typing import Optional, Union, Tuple, List, Dict, Any, Callable

import numpy as np
import keras
from keras import layers, ops

from k3_node.layers.attention.pair_attention import (
    SelfMultiheadAttentionWithPair,
    TriangleMultiplication,
    OuterProduct,
    _get_activation,
)
from k3_node.data.download import download_url


# ==============================================================================
# Model Configurations & Pretrained Weights
# ==============================================================================

UNIMOL2_CONFIGS: Dict[str, Dict[str, Any]] = {
    "84m": {
        "num_encoder_layers": 12,
        "encoder_embed_dim": 768,
        "num_attention_heads": 48,
        "pair_embed_dim": 512,
        "pair_hidden_dim": 64,
        "ffn_embedding_dim": 768,
        "filename": "checkpoint_84m.pt",
        "url": "https://huggingface.co/dptech/Uni-Mol2/resolve/main/modelzoo/84M/checkpoint.pt",
    },
    "164m": {
        "num_encoder_layers": 24,
        "encoder_embed_dim": 768,
        "num_attention_heads": 48,
        "pair_embed_dim": 512,
        "pair_hidden_dim": 64,
        "ffn_embedding_dim": 768,
        "filename": "checkpoint_164m.pt",
        "url": "https://huggingface.co/dptech/Uni-Mol2/resolve/main/modelzoo/164M/checkpoint.pt",
    },
    "310m": {
        "num_encoder_layers": 32,
        "encoder_embed_dim": 1024,
        "num_attention_heads": 64,
        "pair_embed_dim": 512,
        "pair_hidden_dim": 64,
        "ffn_embedding_dim": 1024,
        "filename": "checkpoint_310m.pt",
        "url": "https://huggingface.co/dptech/Uni-Mol2/resolve/main/modelzoo/310M/checkpoint.pt",
    },
    "570m": {
        "num_encoder_layers": 32,
        "encoder_embed_dim": 1536,
        "num_attention_heads": 96,
        "pair_embed_dim": 512,
        "pair_hidden_dim": 64,
        "ffn_embedding_dim": 1536,
        "filename": "checkpoint_570m.pt",
        "url": "https://huggingface.co/dptech/Uni-Mol2/resolve/main/modelzoo/570M/checkpoint.pt",
    },
    "1.1b": {
        "num_encoder_layers": 64,
        "encoder_embed_dim": 1536,
        "num_attention_heads": 96,
        "pair_embed_dim": 512,
        "pair_hidden_dim": 64,
        "ffn_embedding_dim": 1536,
        "filename": "checkpoint_1.1b.pt",
        "url": "https://huggingface.co/dptech/Uni-Mol2/resolve/main/modelzoo/1.1B/checkpoint.pt",
    },
}


# ==============================================================================
# Feature Extraction and Geometry Layers
# ==============================================================================

class AtomFeature(layers.Layer):
    r"""Multi-attribute atomic embedding layer (atom types, formal charges, degrees)."""

    def __init__(
        self,
        num_atom: int = 512,
        num_degree: int = 128,
        num_charge: int = 128,
        embed_dim: int = 768,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.atom_embed = layers.Embedding(num_atom, embed_dim, name="atom_embed")
        self.degree_embed = layers.Embedding(num_degree, embed_dim, name="degree_embed")
        self.charge_embed = layers.Embedding(num_charge, embed_dim, name="charge_embed")
        self.norm = layers.LayerNormalization(axis=-1, epsilon=1e-5, name="norm")

    def build(self, input_shape=None):
        if not self.built:
            self.atom_embed.build(None)
            self.degree_embed.build(None)
            self.charge_embed.build(None)
            self.norm.build((None, None, self.atom_embed.output_dim))
        super().build(input_shape)

    def call(self, atom_types, degrees=None, charges=None):
        x = self.atom_embed(atom_types)
        if degrees is not None:
            x = x + self.degree_embed(degrees)
        if charges is not None:
            x = x + self.charge_embed(charges)
        return self.norm(x)


class EdgeFeature(layers.Layer):
    r"""Pairwise edge feature embedding (bonds, shortest path distance)."""

    def __init__(
        self,
        num_edge: int = 64,
        num_spatial: int = 512,
        pair_dim: int = 512,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.edge_embed = layers.Embedding(num_edge, pair_dim, name="edge_embed")
        self.spatial_embed = layers.Embedding(num_spatial, pair_dim, name="spatial_embed")
        self.norm = layers.LayerNormalization(axis=-1, epsilon=1e-5, name="norm")

    def build(self, input_shape=None):
        if not self.built:
            self.edge_embed.build(None)
            self.spatial_embed.build(None)
            self.norm.build((None, None, None, self.edge_embed.output_dim))
        super().build(input_shape)

    def call(self, edge_types, spatial_types=None):
        pair = self.edge_embed(edge_types)
        if spatial_types is not None:
            pair = pair + self.spatial_embed(spatial_types)
        return self.norm(pair)


class SE3InvariantKernel(layers.Layer):
    r"""SE(3)-invariant geometric kernel combining Gaussian radial distances."""

    def __init__(
        self,
        num_kernel: int = 128,
        pair_dim: int = 512,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_kernel = num_kernel
        self.pair_dim = pair_dim

        self.means = self.add_weight(
            shape=(num_kernel,),
            initializer="uniform",
            trainable=True,
            name="means",
        )
        self.stds = self.add_weight(
            shape=(num_kernel,),
            initializer="uniform",
            trainable=True,
            name="stds",
        )
        self.proj = layers.Dense(pair_dim, name="proj")

    def build(self, input_shape=None):
        if not self.built:
            self.proj.build((None, None, None, self.num_kernel))
        super().build(input_shape)

    def call(self, dist):
        # dist: [B, N, N]
        x = ops.expand_dims(dist, axis=-1)  # [B, N, N, 1]
        mean = self.means
        std = ops.abs(self.stds) + 1e-5

        a = math.sqrt(2.0 * math.pi)
        diff = (x - mean) / std
        gbf = ops.exp(-0.5 * ops.power(diff, 2)) / (a * std)
        return self.proj(gbf)


class MovementPredictionHead(layers.Layer):
    r"""SE(3)-equivariant coordinate movement prediction head for Uni-Mol2."""

    def __init__(self, pair_dim: int, hidden_dim: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.pair_dim = pair_dim
        self.hidden_dim = hidden_dim

        self.norm = layers.LayerNormalization(axis=-1, epsilon=1e-5, name="norm")
        self.dense1 = layers.Dense(hidden_dim, activation="gelu", name="dense1")
        self.dense2 = layers.Dense(1, use_bias=False, name="dense2")

    def build(self, input_shape=None):
        if not self.built:
            self.norm.build((None, None, None, self.pair_dim))
            self.dense1.build((None, None, None, self.pair_dim))
            self.dense2.build((None, None, None, self.hidden_dim))
        super().build(input_shape)

    def call(self, coords, pair, mask=None):
        r"""
        Args:
            coords (Tensor): Node 3D coordinates ``[batch_size, seq_len, 3]``.
            pair (Tensor): Pair representations ``[batch_size, seq_len, seq_len, pair_dim]``.
            mask (Tensor, optional): Optional node mask ``[batch_size, seq_len]``.
        """
        w = self.norm(pair)
        w = self.dense1(w)
        w = self.dense2(w)  # [B, N, N, 1]

        # [B, N, 1, 3] - [B, 1, N, 3] = [B, N, N, 3]
        diff_pos = ops.expand_dims(coords, axis=2) - ops.expand_dims(coords, axis=1)

        if mask is not None:
            m = ops.expand_dims(ops.cast(mask, diff_pos.dtype), axis=-1)
            pair_m = ops.expand_dims(m, axis=2) * ops.expand_dims(m, axis=1)
            diff_pos = diff_pos * pair_m

        delta_pos = ops.sum(diff_pos * w, axis=2)
        return coords + delta_pos


# ==============================================================================
# Two-Track Transformer Layer
# ==============================================================================

class UniMol2TransformerLayer(layers.Layer):
    r"""Two-track transformer layer for Uni-Mol2 (atom track and pair track)."""

    def __init__(
        self,
        embed_dim: int = 768,
        pair_dim: int = 512,
        pair_hidden_dim: int = 64,
        ffn_embed_dim: int = 768,
        num_heads: int = 48,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.pair_dim = pair_dim

        # Pair bias projection to heads
        self.pair_to_heads = layers.Dense(num_heads, name="pair_to_heads")

        # 1. Atom Track
        self.self_attn = SelfMultiheadAttentionWithPair(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            name="self_attn",
        )
        self.attn_norm = layers.LayerNormalization(axis=-1, epsilon=1e-5, name="attn_norm")

        self.atom_ffn1 = layers.Dense(ffn_embed_dim, activation="gelu", name="atom_ffn1")
        self.atom_ffn2 = layers.Dense(embed_dim, name="atom_ffn2")
        self.atom_norm = layers.LayerNormalization(axis=-1, epsilon=1e-5, name="atom_norm")

        # 2. Pair Track
        self.outer_product = OuterProduct(
            embed_dim=embed_dim,
            pair_dim=pair_dim,
            hidden_dim=32,
            name="outer_product",
        )
        self.tri_out = TriangleMultiplication(
            pair_dim=pair_dim,
            hidden_dim=pair_hidden_dim,
            mode="outgoing",
            name="tri_out",
        )
        self.tri_in = TriangleMultiplication(
            pair_dim=pair_dim,
            hidden_dim=pair_hidden_dim,
            mode="incoming",
            name="tri_in",
        )
        self.pair_ffn1 = layers.Dense(pair_dim, activation="gelu", name="pair_ffn1")
        self.pair_ffn2 = layers.Dense(pair_dim, name="pair_ffn2")
        self.pair_norm = layers.LayerNormalization(axis=-1, epsilon=1e-5, name="pair_norm")

    def build(self, input_shape=None):
        if not self.built:
            self.pair_to_heads.build((None, None, None, self.pair_dim))
            self.self_attn.build((None, None, self.embed_dim))
            self.attn_norm.build((None, None, self.embed_dim))
            self.atom_ffn1.build((None, None, self.embed_dim))
            self.atom_ffn2.build((None, None, self.atom_ffn1.units))
            self.atom_norm.build((None, None, self.embed_dim))

            self.outer_product.build((None, None, self.embed_dim))
            self.tri_out.build((None, None, None, self.pair_dim))
            self.tri_in.build((None, None, None, self.pair_dim))
            self.pair_ffn1.build((None, None, None, self.pair_dim))
            self.pair_ffn2.build((None, None, None, self.pair_dim))
            self.pair_norm.build((None, None, None, self.pair_dim))
        super().build(input_shape)

    def call(self, x, pair, padding_mask=None, training: bool = False):
        # 1. Update pair track with outer product
        pair = pair + self.outer_product(x, training=training)
        pair = self.tri_out(pair, training=training)
        pair = self.tri_in(pair, training=training)
        pair_res = pair
        pair = self.pair_norm(pair)
        pair = pair_res + self.pair_ffn2(self.pair_ffn1(pair))

        # Project pair representation to attention heads: [B, N, N, H] -> [B, H, N, N]
        attn_bias = self.pair_to_heads(pair)

        # 2. Update atom track with pair-augmented attention
        attn_out = self.self_attn(
            query=x,
            key_padding_mask=padding_mask,
            attn_bias=attn_bias,
            training=training,
        )
        x = self.attn_norm(x + attn_out)
        atom_res = x
        x = self.atom_norm(atom_res + self.atom_ffn2(self.atom_ffn1(x)))

        return x, pair


# ==============================================================================
# Full Uni-Mol2 Model
# ==============================================================================

class UniMol2Model(keras.Model):
    r"""Scalable molecular pretraining model Uni-Mol2 (84M to 1.1B parameters).

    Args:
        model_size (str, optional): Model scale preset (``"84m"``, ``"164m"``, ``"310m"``,
            ``"570m"``, or ``"1.1b"``). (default: ``"84m"``)
        output_dim (int, optional): Classification/regression head dimension. (default: ``2``)
        num_encoder_layers (int, optional): Number of two-track transformer layers.
        encoder_embed_dim (int, optional): Atom feature dimension.
        num_attention_heads (int, optional): Number of attention heads.
        pair_embed_dim (int, optional): Pair feature dimension.
        **kwargs: Additional model arguments.
    """

    def __init__(
        self,
        model_size: str = "84m",
        output_dim: int = 2,
        num_encoder_layers: Optional[int] = None,
        encoder_embed_dim: Optional[int] = None,
        num_attention_heads: Optional[int] = None,
        pair_embed_dim: Optional[int] = None,
        ffn_embedding_dim: Optional[int] = None,
        pair_hidden_dim: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        clean_size = model_size.lower()
        if clean_size not in UNIMOL2_CONFIGS:
            clean_size = "84m"

        cfg = UNIMOL2_CONFIGS[clean_size]
        self.model_size = clean_size
        self.output_dim = output_dim
        self.num_layers = num_encoder_layers or cfg["num_encoder_layers"]
        self.embed_dim = encoder_embed_dim or cfg["encoder_embed_dim"]
        self.num_heads = num_attention_heads or cfg["num_attention_heads"]
        self.pair_dim = pair_embed_dim or cfg["pair_embed_dim"]
        self.ffn_dim = ffn_embedding_dim or cfg["ffn_embedding_dim"]
        self.pair_hidden_dim = pair_hidden_dim or cfg["pair_hidden_dim"]

        self.atom_feature = AtomFeature(embed_dim=self.embed_dim, name="atom_feature")
        self.edge_feature = EdgeFeature(pair_dim=self.pair_dim, name="edge_feature")
        self.se3_kernel = SE3InvariantKernel(pair_dim=self.pair_dim, name="se3_kernel")

        self.layers_list = [
            UniMol2TransformerLayer(
                embed_dim=self.embed_dim,
                pair_dim=self.pair_dim,
                pair_hidden_dim=self.pair_hidden_dim,
                ffn_embed_dim=self.ffn_dim,
                num_heads=self.num_heads,
                name=f"layer_{i}",
            )
            for i in range(self.num_layers)
        ]

        self.movement_head = MovementPredictionHead(pair_dim=self.pair_dim, name="movement_head")
        self.head = layers.Dense(output_dim, name="head")

    def build(self, input_shape=None):
        if not self.built:
            self.atom_feature.build(None)
            self.edge_feature.build(None)
            self.se3_kernel.build(None)
            for l in self.layers_list:
                l.build(None)
            self.movement_head.build(None)
            self.head.build((None, self.embed_dim))
        super().build(input_shape)

    def call(
        self,
        atom_types,
        coords=None,
        edge_types=None,
        degrees=None,
        charges=None,
        padding_mask=None,
        return_coords: bool = False,
        return_repr: bool = False,
        training: bool = False,
    ):
        if isinstance(atom_types, dict):
            coords = atom_types.get("coords", coords)
            edge_types = atom_types.get("edge_types", edge_types)
            degrees = atom_types.get("degrees", degrees)
            charges = atom_types.get("charges", charges)
            padding_mask = atom_types.get("padding_mask", padding_mask)
            atom_types = atom_types.get("atom_types", atom_types.get("tokens"))
        elif isinstance(atom_types, (tuple, list)):
            if len(atom_types) == 2:
                atom_types, coords = atom_types

        shape = ops.shape(atom_types)
        bsz = shape[0]
        seq_len = shape[1]

        if coords is None:
            coords = ops.zeros((bsz, seq_len, 3), dtype="float32")

        if edge_types is None:
            edge_types = ops.cast(ops.mod(ops.expand_dims(atom_types, axis=-1) * 8 + ops.expand_dims(atom_types, axis=1), 64), "int32")

        # 1. Feature representations
        x = self.atom_feature(atom_types, degrees=degrees, charges=charges)
        pair = self.edge_feature(edge_types)

        # Distance geometry
        diff = ops.expand_dims(coords, axis=2) - ops.expand_dims(coords, axis=1)
        dist = ops.sqrt(ops.sum(ops.power(diff, 2), axis=-1) + 1e-10)
        pair = pair + self.se3_kernel(dist)

        # 2. Transformer layers
        for enc_layer in self.layers_list:
            x, pair = enc_layer(x, pair, padding_mask=padding_mask, training=training)

        # Updated coordinates
        updated_coords = self.movement_head(coords, pair, mask=1.0 - ops.cast(padding_mask, "float32") if padding_mask is not None else None)

        cls_rep = x[:, 0, :]
        logits = self.head(cls_rep)

        if return_coords and return_repr:
            return logits, updated_coords, x, pair
        if return_coords:
            return logits, updated_coords
        if return_repr:
            return logits, x, pair
        return logits


# ==============================================================================
# Helper Functions: Download & Load Weights
# ==============================================================================

def download_unimol2_checkpoint(
    model_size: str = "84m",
    folder: str = "checkpoints",
    log: bool = True,
) -> str:
    r"""Downloads a pre-trained Uni-Mol2 checkpoint (.pt).

    Args:
        model_size (str): Model size (``"84m"``, ``"164m"``, ``"310m"``, ``"570m"``, ``"1.1b"``).
        folder (str, optional): Target folder. (default: ``"checkpoints"``)
        log (bool, optional): Print download progress. (default: ``True``)

    Returns:
        str: Absolute path to the downloaded file.
    """
    clean_size = model_size.lower().strip()
    if clean_size not in UNIMOL2_CONFIGS:
        raise ValueError(f"Unknown Uni-Mol2 size '{model_size}'. Available: {list(UNIMOL2_CONFIGS.keys())}")

    info = UNIMOL2_CONFIGS[clean_size]
    local_path = os.path.join(folder, info["filename"])
    if os.path.exists(local_path):
        return local_path

    # Check Uni-Mol repo fallback
    alt_local = os.path.join("Uni-Mol", "unimol2", "modelzoo", clean_size.upper(), "checkpoint.pt")
    if os.path.exists(alt_local):
        return alt_local

    return download_url(info["url"], folder=folder, filename=info["filename"], log=log)


def load_unimol2_weights(
    model: UniMol2Model,
    checkpoint_path: Optional[str] = None,
    model_size: Optional[str] = None,
    folder: str = "checkpoints",
    download: bool = True,
) -> UniMol2Model:
    r"""Loads weights from a PyTorch checkpoint into a Keras 3 UniMol2Model."""
    path_to_load = checkpoint_path

    size = model_size or model.model_size
    if path_to_load is None:
        if size in UNIMOL2_CONFIGS:
            local_target = os.path.join(folder, UNIMOL2_CONFIGS[size]["filename"])
            if os.path.isfile(local_target):
                path_to_load = local_target
            elif download:
                path_to_load = download_unimol2_checkpoint(size, folder=folder)
            else:
                raise FileNotFoundError(f"Checkpoint for '{size}' not found at '{local_target}'.")
        else:
            raise ValueError(f"Unknown model size '{size}'.")
    elif not os.path.isfile(path_to_load) and download and size in UNIMOL2_CONFIGS:
        path_to_load = download_unimol2_checkpoint(size, folder=folder)

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
        raise ValueError(f"Expected dict in checkpoint, got {type(state)}")

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

    # Embeddings
    if "atom_feature.atom_embed.weight" in state_dict:
        model.atom_feature.atom_embed.embeddings.assign(_to_tensor(state_dict["atom_feature.atom_embed.weight"]))
    if "edge_feature.edge_embed.weight" in state_dict:
        model.edge_feature.edge_embed.embeddings.assign(_to_tensor(state_dict["edge_feature.edge_embed.weight"]))

    return model
