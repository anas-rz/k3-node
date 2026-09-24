import os
import tempfile
import numpy as np
import pytest
import torch
import keras
from keras import ops

from k3_node.models import (
    UniMol2Model,
    UniMol2AtomFeature,
    UniMol2EdgeFeature,
    UniMol2SE3Kernel,
    UniMol2MovementHead,
    download_unimol2_checkpoint,
    load_unimol2_weights,
)


def test_unimol2_features():
    bsz, seq_len = 2, 5
    embed_dim = 32
    pair_dim = 16

    atom_feature = UniMol2AtomFeature(num_atom=128, embed_dim=embed_dim)
    tokens = ops.convert_to_tensor(np.random.randint(0, 128, (bsz, seq_len)), dtype="int32")
    x = atom_feature(tokens)
    assert ops.shape(x) == (bsz, seq_len, embed_dim)

    edge_feature = UniMol2EdgeFeature(num_edge=32, pair_dim=pair_dim)
    edge_types = ops.convert_to_tensor(np.random.randint(0, 32, (bsz, seq_len, seq_len)), dtype="int32")
    pair = edge_feature(edge_types)
    assert ops.shape(pair) == (bsz, seq_len, seq_len, pair_dim)

    se3 = UniMol2SE3Kernel(num_kernel=32, pair_dim=pair_dim)
    dist = ops.convert_to_tensor(np.random.uniform(0.5, 5.0, (bsz, seq_len, seq_len)).astype("float32"))
    pair_geo = se3(dist)
    assert ops.shape(pair_geo) == (bsz, seq_len, seq_len, pair_dim)


def test_unimol2_movement_head():
    bsz, seq_len = 2, 6
    pair_dim = 16
    coords = ops.convert_to_tensor(np.random.randn(bsz, seq_len, 3).astype("float32"))
    pair = ops.convert_to_tensor(np.random.randn(bsz, seq_len, seq_len, pair_dim).astype("float32"))

    head = UniMol2MovementHead(pair_dim=pair_dim, hidden_dim=32)
    updated = head(coords, pair)
    assert ops.shape(updated) == (bsz, seq_len, 3)


def test_unimol2_model_forward():
    bsz, seq_len = 2, 6
    embed_dim = 32
    pair_dim = 16
    num_heads = 4
    num_layers = 2

    model = UniMol2Model(
        model_size="84m",
        output_dim=2,
        num_encoder_layers=num_layers,
        encoder_embed_dim=embed_dim,
        num_attention_heads=num_heads,
        pair_embed_dim=pair_dim,
        ffn_embedding_dim=64,
        pair_hidden_dim=8,
    )

    atom_types = ops.convert_to_tensor(np.random.randint(0, 128, (bsz, seq_len)), dtype="int32")
    coords = ops.convert_to_tensor(np.random.randn(bsz, seq_len, 3).astype("float32"))

    # Logits
    logits = model(atom_types, coords=coords)
    assert ops.shape(logits) == (bsz, 2)

    # Return coords and representations
    logits, new_coords, x, pair = model(atom_types, coords=coords, return_coords=True, return_repr=True)
    assert ops.shape(logits) == (bsz, 2)
    assert ops.shape(new_coords) == (bsz, seq_len, 3)
    assert ops.shape(x) == (bsz, seq_len, embed_dim)
    assert ops.shape(pair) == (bsz, seq_len, seq_len, pair_dim)


def test_unimol2_weight_loading():
    model = UniMol2Model(
        model_size="84m",
        output_dim=2,
        num_encoder_layers=1,
        encoder_embed_dim=16,
        num_attention_heads=2,
        pair_embed_dim=8,
        ffn_embedding_dim=16,
        pair_hidden_dim=4,
    )
    model.build(None)

    state_dict = {
        "atom_feature.atom_embed.weight": torch.randn(512, 16),
        "edge_feature.edge_embed.weight": torch.randn(64, 8),
    }

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = tmp.name
        torch.save({"model": state_dict}, tmp_path)

    try:
        load_unimol2_weights(model, checkpoint_path=tmp_path, download=False)
        w = ops.convert_to_numpy(model.atom_feature.atom_embed.embeddings)
        np.testing.assert_allclose(w, state_dict["atom_feature.atom_embed.weight"].numpy(), atol=1e-5)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

