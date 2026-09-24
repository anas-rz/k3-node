import math
import os
from typing import Optional, Union, Tuple, List, Dict, Any, Callable

import numpy as np
import keras
from keras import layers, ops

from k3_node.layers.attention.pair_attention import (
    SelfMultiheadAttentionWithPair,
    TransformerEncoderLayerWithPair,
    TriangleMultiplication,
    OuterProduct,
    _get_activation,
)
from k3_node.models.unimol import (
    GaussianLayer,
    NonLinearHead,
    DistanceHead,
)
from k3_node.data.download import download_url


# ==============================================================================
# Pretrained Weights Registry
# ==============================================================================

DOCKING_V2_CHECKPOINTS: Dict[str, Dict[str, str]] = {
    "v2": {
        "url": "https://huggingface.co/dptech/Uni-Mol-Models/resolve/main/unimol_docking_v2_240517.pt",
        "alt_url": "https://www.dropbox.com/scl/fi/sfhrtx1tjprce18wbvmdr/unimol_docking_v2_240517.pt?rlkey=5zg7bh150kcinalrqdhzmyyoo&st=n6j0nt6c&dl=1",
        "filename": "unimol_docking_v2_240517.pt",
        "description": "Uni-Mol Docking V2 model for realistic and accurate protein-ligand binding pose prediction",
    }
}


# ==============================================================================
# Model
# ==============================================================================

class DockingPoseModelV2(keras.Model):
    r"""Uni-Mol Docking V2 model for joint protein pocket and ligand holo binding pose prediction.

    Args:
        mol_vocab_size (int, optional): Molecule vocabulary size. (default: ``512``)
        pocket_vocab_size (int, optional): Pocket residue vocabulary size. (default: ``512``)
        embed_dim (int, optional): Transformer feature embedding dimension. (default: ``512``)
        pair_dim (int, optional): Pairwise feature dimension. (default: ``128``)
        num_layers (int, optional): Number of joint transformer layers. (default: ``12``)
        num_heads (int, optional): Number of multihead attention heads. (default: ``32``)
        **kwargs: Additional model arguments.
    """

    def __init__(
        self,
        mol_vocab_size: int = 512,
        pocket_vocab_size: int = 512,
        embed_dim: int = 512,
        pair_dim: int = 128,
        num_layers: int = 12,
        num_heads: int = 32,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mol_vocab_size = mol_vocab_size
        self.pocket_vocab_size = pocket_vocab_size
        self.embed_dim = embed_dim
        self.pair_dim = pair_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.mol_embed = layers.Embedding(mol_vocab_size, embed_dim, name="mol_embed")
        self.pocket_embed = layers.Embedding(pocket_vocab_size, embed_dim, name="pocket_embed")

        self.gbf = GaussianLayer(num_kernel=pair_dim, edge_types=1024, name="gbf")
        self.gbf_proj = NonLinearHead(pair_dim, num_heads, activation_fn="gelu", name="gbf_proj")

        self.layers_list = [
            TransformerEncoderLayerWithPair(
                embed_dim=embed_dim,
                ffn_embed_dim=embed_dim * 4,
                attention_heads=num_heads,
                activation_fn="gelu",
                name=f"layer_{i}",
            )
            for i in range(num_layers)
        ]

        self.pose_head = NonLinearHead(num_heads, 1, activation_fn="gelu", name="pose_head")
        self.dist_head = DistanceHead(num_heads, activation_fn="gelu", name="dist_head")

    def build(self, input_shape=None):
        if not self.built:
            self.mol_embed.build(None)
            self.pocket_embed.build(None)
            self.gbf.build(None)
            self.gbf_proj.build((None, None, None, self.pair_dim))
            for layer in self.layers_list:
                layer.build(None)
            self.pose_head.build((None, None, None, self.num_heads))
            self.dist_head.build((None, None, None, self.num_heads))
        super().build(input_shape)

    def call(
        self,
        mol_tokens,
        pocket_tokens=None,
        mol_coords=None,
        pocket_coords=None,
        mol_mask=None,
        pocket_mask=None,
        training: bool = False,
    ):
        r"""Forward pass for Docking V2.

        Args:
            mol_tokens (Tensor): Ligand tokens ``[batch_size, mol_len]``.
            pocket_tokens (Tensor): Pocket tokens ``[batch_size, pocket_len]``.
            mol_coords (Tensor, optional): Initial ligand 3D coordinates ``[batch_size, mol_len, 3]``.
            pocket_coords (Tensor, optional): Pocket 3D coordinates ``[batch_size, pocket_len, 3]``.
            mol_mask (Tensor, optional): Boolean padding mask for ligand.
            pocket_mask (Tensor, optional): Boolean padding mask for pocket.
            training (bool, optional): Training flag.

        Returns:
            Tuple[Tensor, Tensor]: Predicted docked ligand 3D coordinates and predicted complex distance matrix.
        """

        if isinstance(mol_tokens, dict):
            pocket_tokens = mol_tokens.get("pocket_tokens", pocket_tokens)
            mol_coords = mol_tokens.get("mol_coords", mol_coords)
            pocket_coords = mol_tokens.get("pocket_coords", pocket_coords)
            mol_mask = mol_tokens.get("mol_mask", mol_mask)
            pocket_mask = mol_tokens.get("pocket_mask", pocket_mask)
            mol_tokens = mol_tokens.get("mol_tokens")
        elif isinstance(mol_tokens, (tuple, list)):
            if len(mol_tokens) == 4:
                mol_tokens, pocket_tokens, mol_coords, pocket_coords = mol_tokens
            elif len(mol_tokens) == 2:
                mol_tokens, pocket_tokens = mol_tokens

        shape_mol = ops.shape(mol_tokens)
        shape_pkt = ops.shape(pocket_tokens)

        bsz = shape_mol[0]
        n_mol = shape_mol[1]
        n_pkt = shape_pkt[1]
        total_len = n_mol + n_pkt

        if mol_coords is None:
            mol_coords = ops.zeros((bsz, n_mol, 3), dtype="float32")
        if pocket_coords is None:
            pocket_coords = ops.zeros((bsz, n_pkt, 3), dtype="float32")

        # Concatenate tokens and coordinates
        tokens_all = ops.concatenate([mol_tokens, pocket_tokens], axis=1)
        coords_all = ops.concatenate([mol_coords, pocket_coords], axis=1)

        x_mol = self.mol_embed(mol_tokens)
        x_pkt = self.pocket_embed(pocket_tokens)
        x = ops.concatenate([x_mol, x_pkt], axis=1)

        # Distance geometry
        diff = ops.expand_dims(coords_all, axis=2) - ops.expand_dims(coords_all, axis=1)
        dist = ops.sqrt(ops.sum(ops.power(diff, 2), axis=-1) + 1e-10)

        edge_types = ops.cast(ops.mod(ops.expand_dims(tokens_all, axis=-1) * 32 + ops.expand_dims(tokens_all, axis=1), 1024), "int32")
        gbf_feature = self.gbf(dist, edge_types)
        attn_bias = self.gbf_proj(gbf_feature)
        attn_bias = ops.transpose(attn_bias, (0, 3, 1, 2))

        padding_mask = None
        if mol_mask is not None and pocket_mask is not None:
            padding_mask = ops.concatenate([mol_mask, pocket_mask], axis=1)

        curr_bias = attn_bias
        for layer in self.layers_list:
            x, curr_bias, _ = layer(x, attn_bias=curr_bias, padding_mask=padding_mask, return_attn=True, training=training)

        delta_pair = curr_bias - attn_bias
        pair_shape = ops.shape(curr_bias)
        if len(pair_shape) == 3:
            curr_bias = ops.reshape(curr_bias, (bsz, self.num_heads, total_len, total_len))
            delta_pair = ops.reshape(delta_pair, (bsz, self.num_heads, total_len, total_len))

        curr_bias = ops.transpose(curr_bias, (0, 2, 3, 1))
        delta_pair = ops.transpose(delta_pair, (0, 2, 3, 1))

        # Coordinate prediction for ligand
        probs = self.pose_head(delta_pair)
        diff_pos = ops.expand_dims(coords_all, axis=1) - ops.expand_dims(coords_all, axis=2)
        coord_update = coords_all + ops.sum(diff_pos * probs, axis=2)

        pred_mol_coords = coord_update[:, :n_mol, :]
        pred_dist = self.dist_head(curr_bias)
        return pred_mol_coords, pred_dist


# ==============================================================================
# Helper Functions: Download & Load Weights
# ==============================================================================

def download_unimol_docking_checkpoint(
    version: str = "v2",
    folder: str = "checkpoints",
    log: bool = True,
) -> str:
    r"""Downloads a pre-trained Uni-Mol Docking V2 checkpoint (.pt).

    Args:
        version (str): Docking model version (``"v2"``).
        folder (str, optional): Target download folder. (default: ``"checkpoints"``)
        log (bool, optional): Print download log. (default: ``True``)

    Returns:
        str: Absolute path to the downloaded file.
    """
    clean = version.strip().lower()
    if clean not in DOCKING_V2_CHECKPOINTS:
        clean = "v2"

    info = DOCKING_V2_CHECKPOINTS[clean]
    local_path = os.path.join(folder, info["filename"])
    if os.path.exists(local_path):
        return local_path

    # Local fallback
    alt_local = os.path.join("Uni-Mol", "unimol_docking_v2", info["filename"])
    if os.path.exists(alt_local):
        return alt_local

    return download_url(info["url"], folder=folder, filename=info["filename"], log=log)


def load_unimol_docking_weights(
    model: DockingPoseModelV2,
    checkpoint_path: Optional[str] = None,
    folder: str = "checkpoints",
    download: bool = True,
) -> DockingPoseModelV2:
    r"""Loads pre-trained weights into a Keras 3 DockingPoseModelV2 model."""
    path_to_load = checkpoint_path

    if path_to_load is None:
        local_target = os.path.join(folder, DOCKING_V2_CHECKPOINTS["v2"]["filename"])
        if os.path.isfile(local_target):
            path_to_load = local_target
        elif download:
            path_to_load = download_unimol_docking_checkpoint("v2", folder=folder)
        else:
            raise FileNotFoundError(f"Checkpoint not found at '{local_target}'.")
    elif not os.path.isfile(path_to_load) and download:
        path_to_load = download_unimol_docking_checkpoint("v2", folder=folder)

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

    if "mol_embed.weight" in state_dict:
        model.mol_embed.embeddings.assign(_to_tensor(state_dict["mol_embed.weight"]))
    if "pocket_embed.weight" in state_dict:
        model.pocket_embed.embeddings.assign(_to_tensor(state_dict["pocket_embed.weight"]))

    return model
