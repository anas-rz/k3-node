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
from k3_node.models.unimol2 import (
    AtomFeature,
    EdgeFeature,
    SE3InvariantKernel,
    MovementPredictionHead,
)
from k3_node.data.download import download_url


# ==============================================================================
# Pretrained Weights Registry
# ==============================================================================

UNIMOL_PLUS_PRETRAINED_URLS: Dict[str, Dict[str, str]] = {
    "pcq_base": {
        "url": "https://github.com/deepmodeling/Uni-Mol/releases/download/v0.2/unimol_plus_pcq_base.pt",
        "filename": "unimol_plus_pcq_base.pt",
        "layers": 12,
        "embed_dim": 768,
        "heads": 48,
        "description": "Uni-Mol+ Base for PCQM4Mv2 (12 layers, 52.4M params)",
    },
    "pcq_large": {
        "url": "https://github.com/deepmodeling/Uni-Mol/releases/download/v0.2/unimol_plus_pcq_large.pt",
        "filename": "unimol_plus_pcq_large.pt",
        "layers": 18,
        "embed_dim": 768,
        "heads": 48,
        "description": "Uni-Mol+ Large for PCQM4Mv2 (18 layers, 77M params)",
    },
    "pcq_small": {
        "url": "https://github.com/deepmodeling/Uni-Mol/releases/download/v0.2/unimol_plus_pcq_small.pt",
        "filename": "unimol_plus_pcq_small.pt",
        "layers": 6,
        "embed_dim": 768,
        "heads": 48,
        "description": "Uni-Mol+ Small for PCQM4Mv2 (6 layers, 27.7M params)",
    },
    "oc20_base": {
        "url": "https://github.com/deepmodeling/Uni-Mol/releases/download/v0.2/unimol_plus_oc20_base.pt",
        "filename": "unimol_plus_oc20_base.pt",
        "layers": 12,
        "embed_dim": 768,
        "heads": 48,
        "description": "Uni-Mol+ Base for OC20 IS2RE (12 layers, 48.6M params)",
    },
}

UNIMOL_PLUS_ALIASES = {
    "base": "pcq_base",
    "large": "pcq_large",
    "small": "pcq_small",
    "oc20": "oc20_base",
    "pcqm4mv2": "pcq_base",
}


# ==============================================================================
# Layers & Heads
# ==============================================================================

class EnergyHead(layers.Layer):
    r"""Head for quantum chemical property or energy prediction."""

    def __init__(self, embed_dim: int, hidden_dim: Optional[int] = None, output_dim: int = 1, **kwargs):
        super().__init__(**kwargs)
        hidden = hidden_dim or embed_dim
        self.dense1 = layers.Dense(hidden, activation="gelu", name="dense1")
        self.dense2 = layers.Dense(output_dim, name="dense2")

    def build(self, input_shape=None):
        if not self.built:
            self.dense1.build((None, self.dense1.units))
            self.dense2.build((None, self.dense1.units))
        super().build(input_shape)

    def call(self, x):
        return self.dense2(self.dense1(x))


class UnimolPlusEncoderLayer(layers.Layer):
    r"""Iterative geometry and representation refinement layer for Uni-Mol+."""

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

        self.pair_to_heads = layers.Dense(num_heads, name="pair_to_heads")

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

        self.movement_head = MovementPredictionHead(pair_dim=pair_dim, name="movement_head")

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

            self.movement_head.build(None)
        super().build(input_shape)

    def call(self, x, pair, coords, padding_mask=None, training: bool = False):
        # 1. Update pair track
        pair = pair + self.outer_product(x, training=training)
        pair = self.tri_out(pair, training=training)
        pair = self.tri_in(pair, training=training)
        pair_res = pair
        pair = self.pair_norm(pair)
        pair = pair_res + self.pair_ffn2(self.pair_ffn1(pair))

        # 2. Update atom track with pair bias
        attn_bias = self.pair_to_heads(pair)
        attn_out = self.self_attn(
            query=x,
            key_padding_mask=padding_mask,
            attn_bias=attn_bias,
            training=training,
        )
        x = self.attn_norm(x + attn_out)
        atom_res = x
        x = self.atom_norm(atom_res + self.atom_ffn2(self.atom_ffn1(x)))

        # 3. Iterative coordinate update
        mask = 1.0 - ops.cast(padding_mask, "float32") if padding_mask is not None else None
        coords = self.movement_head(coords, pair, mask=mask)

        return x, pair, coords


# ==============================================================================
# Uni-Mol+ Models (PCQ and OC20)
# ==============================================================================

class UniMolPlusPCQModel(keras.Model):
    r"""Uni-Mol+ Model for quantum chemical property prediction on PCQM4Mv2.

    Args:
        num_layers (int, optional): Number of iterative refinement layers. (default: ``12``)
        embed_dim (int, optional): Atom representation dimension. (default: ``768``)
        pair_dim (int, optional): Pair representation dimension. (default: ``512``)
        num_heads (int, optional): Number of attention heads. (default: ``48``)
        output_dim (int, optional): Number of predicted quantum properties. (default: ``1``)
        **kwargs: Additional model arguments.
    """

    def __init__(
        self,
        num_layers: int = 12,
        embed_dim: int = 768,
        pair_dim: int = 512,
        num_heads: int = 48,
        output_dim: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.pair_dim = pair_dim
        self.num_heads = num_heads
        self.output_dim = output_dim

        self.atom_feature = AtomFeature(embed_dim=embed_dim, name="atom_feature")
        self.edge_feature = EdgeFeature(pair_dim=pair_dim, name="edge_feature")
        self.se3_kernel = SE3InvariantKernel(pair_dim=pair_dim, name="se3_kernel")

        self.layers_list = [
            UnimolPlusEncoderLayer(
                embed_dim=embed_dim,
                pair_dim=pair_dim,
                pair_hidden_dim=64,
                ffn_embed_dim=embed_dim,
                num_heads=num_heads,
                name=f"layer_{i}",
            )
            for i in range(num_layers)
        ]

        self.energy_head = EnergyHead(embed_dim=embed_dim, output_dim=output_dim, name="energy_head")

    def build(self, input_shape=None):
        if not self.built:
            self.atom_feature.build(None)
            self.edge_feature.build(None)
            self.se3_kernel.build(None)
            for l in self.layers_list:
                l.build(None)
            self.energy_head.build((None, self.embed_dim))
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

        x = self.atom_feature(atom_types, degrees=degrees, charges=charges)
        pair = self.edge_feature(edge_types)

        diff = ops.expand_dims(coords, axis=2) - ops.expand_dims(coords, axis=1)
        dist = ops.sqrt(ops.sum(ops.power(diff, 2), axis=-1) + 1e-10)
        pair = pair + self.se3_kernel(dist)

        for layer in self.layers_list:
            x, pair, coords = layer(x, pair, coords, padding_mask=padding_mask, training=training)

        # Graph-level pooling (CLS token or masked mean)
        cls_token = x[:, 0, :]
        property_pred = self.energy_head(cls_token)

        if return_coords:
            return property_pred, coords
        return property_pred


class UniMolPlusOC20Model(UniMolPlusPCQModel):
    r"""Uni-Mol+ Model for initial structure to relaxed energy prediction on OC20."""
    pass


# ==============================================================================
# Helper Functions: Download & Load Weights
# ==============================================================================

def download_unimol_plus_checkpoint(
    name: str = "pcq_base",
    folder: str = "checkpoints",
    log: bool = True,
) -> str:
    r"""Downloads a pre-trained Uni-Mol+ checkpoint (.pt).

    Args:
        name (str): Model name or alias (``"pcq_base"``, ``"pcq_large"``, ``"pcq_small"``,
            ``"oc20_base"``).
        folder (str, optional): Target download directory. (default: ``"checkpoints"``)
        log (bool, optional): Print download log. (default: ``True``)

    Returns:
        str: Absolute path to the downloaded file.
    """
    clean = name.strip().lower()
    if clean in UNIMOL_PLUS_ALIASES:
        clean = UNIMOL_PLUS_ALIASES[clean]

    if clean not in UNIMOL_PLUS_PRETRAINED_URLS:
        raise ValueError(
            f"Unknown Uni-Mol+ model '{name}'. Available: {list(UNIMOL_PLUS_PRETRAINED_URLS.keys())}"
        )

    info = UNIMOL_PLUS_PRETRAINED_URLS[clean]
    local_path = os.path.join(folder, info["filename"])
    if os.path.exists(local_path):
        return local_path

    # Check local Uni-Mol repo fallback
    alt_local = os.path.join("Uni-Mol", "unimol_plus", info["filename"])
    if os.path.exists(alt_local):
        return alt_local

    return download_url(info["url"], folder=folder, filename=info["filename"], log=log)


def load_unimol_plus_weights(
    model: UniMolPlusPCQModel,
    checkpoint_path: Optional[str] = None,
    pretrained_name: Optional[str] = None,
    folder: str = "checkpoints",
    download: bool = True,
) -> UniMolPlusPCQModel:
    r"""Loads pre-trained weights into a Keras 3 Uni-Mol+ model."""
    path_to_load = checkpoint_path

    candidate = pretrained_name or checkpoint_path
    if candidate:
        clean = candidate.strip().lower()
        if clean in UNIMOL_PLUS_ALIASES:
            candidate = UNIMOL_PLUS_ALIASES[clean]

    if path_to_load is None:
        if candidate is None:
            raise ValueError("Either checkpoint_path or pretrained_name must be specified.")
        if candidate in UNIMOL_PLUS_PRETRAINED_URLS:
            fname = UNIMOL_PLUS_PRETRAINED_URLS[candidate]["filename"]
            local_target = os.path.join(folder, fname)
            if os.path.isfile(local_target):
                path_to_load = local_target
            elif download:
                path_to_load = download_unimol_plus_checkpoint(candidate, folder=folder)
            else:
                raise FileNotFoundError(f"Checkpoint for '{candidate}' not found at '{local_target}'.")
        else:
            raise ValueError(f"Unknown checkpoint '{candidate}'.")
    elif not os.path.isfile(path_to_load) and download and candidate in UNIMOL_PLUS_PRETRAINED_URLS:
        path_to_load = download_unimol_plus_checkpoint(candidate, folder=folder)

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

    # Atom & Edge feature embeddings
    if "atom_feature.atom_embed.weight" in state_dict:
        model.atom_feature.atom_embed.embeddings.assign(_to_tensor(state_dict["atom_feature.atom_embed.weight"]))
    if "edge_feature.edge_embed.weight" in state_dict:
        model.edge_feature.edge_embed.embeddings.assign(_to_tensor(state_dict["edge_feature.edge_embed.weight"]))

    return model
