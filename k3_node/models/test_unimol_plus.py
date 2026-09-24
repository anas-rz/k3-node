import os
import tempfile
import numpy as np
import pytest
import torch
import keras
from keras import ops

from k3_node.models import (
    UniMolPlusPCQModel,
    UniMolPlusOC20Model,
    DockingPoseModelV2,
    download_unimol_plus_checkpoint,
    load_unimol_plus_weights,
    download_unimol_docking_checkpoint,
    load_unimol_docking_weights,
)


def test_unimol_plus_pcq_model():
    bsz = 2
    seq_len = 5
    embed_dim = 32
    pair_dim = 16
    num_heads = 4
    num_layers = 2

    model = UniMolPlusPCQModel(
        num_layers=num_layers,
        embed_dim=embed_dim,
        pair_dim=pair_dim,
        num_heads=num_heads,
        output_dim=1,
    )

    atom_types = ops.convert_to_tensor(np.random.randint(0, 64, (bsz, seq_len)), dtype="int32")
    coords = ops.convert_to_tensor(np.random.randn(bsz, seq_len, 3).astype("float32"))

    # Property prediction
    pred = model(atom_types, coords=coords)
    assert ops.shape(pred) == (bsz, 1)

    # Return coords
    pred, new_coords = model(atom_types, coords=coords, return_coords=True)
    assert ops.shape(pred) == (bsz, 1)
    assert ops.shape(new_coords) == (bsz, seq_len, 3)


def test_unimol_plus_oc20_model():
    bsz = 2
    seq_len = 6
    embed_dim = 32
    pair_dim = 16
    num_heads = 4
    num_layers = 2

    model = UniMolPlusOC20Model(
        num_layers=num_layers,
        embed_dim=embed_dim,
        pair_dim=pair_dim,
        num_heads=num_heads,
        output_dim=1,
    )

    atom_types = ops.convert_to_tensor(np.random.randint(0, 64, (bsz, seq_len)), dtype="int32")
    coords = ops.convert_to_tensor(np.random.randn(bsz, seq_len, 3).astype("float32"))

    pred = model(atom_types, coords=coords)
    assert ops.shape(pred) == (bsz, 1)


def test_unimol_docking_v2():
    bsz = 2
    n_mol = 4
    n_pkt = 6
    embed_dim = 32
    pair_dim = 16
    num_heads = 4
    num_layers = 2

    model = DockingPoseModelV2(
        mol_vocab_size=64,
        pocket_vocab_size=64,
        embed_dim=embed_dim,
        pair_dim=pair_dim,
        num_layers=num_layers,
        num_heads=num_heads,
    )

    mol_tokens = ops.convert_to_tensor(np.random.randint(0, 64, (bsz, n_mol)), dtype="int32")
    pkt_tokens = ops.convert_to_tensor(np.random.randint(0, 64, (bsz, n_pkt)), dtype="int32")
    mol_coords = ops.convert_to_tensor(np.random.randn(bsz, n_mol, 3).astype("float32"))
    pkt_coords = ops.convert_to_tensor(np.random.randn(bsz, n_pkt, 3).astype("float32"))

    docked_coords, pred_dist = model(
        mol_tokens,
        pkt_tokens,
        mol_coords=mol_coords,
        pocket_coords=pkt_coords,
    )
    assert ops.shape(docked_coords) == (bsz, n_mol, 3)
    assert ops.shape(pred_dist) == (bsz, n_mol + n_pkt, n_mol + n_pkt)


def test_unimol_plus_weight_loading():
    model = UniMolPlusPCQModel(
        num_layers=1,
        embed_dim=16,
        pair_dim=8,
        num_heads=2,
        output_dim=1,
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
        load_unimol_plus_weights(model, checkpoint_path=tmp_path, download=False)
        w = ops.convert_to_numpy(model.atom_feature.atom_embed.embeddings)
        np.testing.assert_allclose(w, state_dict["atom_feature.atom_embed.weight"].numpy(), atol=1e-5)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

