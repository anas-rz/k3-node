"""Unit tests for multi-backend materials models."""

import pytest
import numpy as np
import keras
from keras import ops

from k3_node.models.materials import (
    MEGNet,
    M3GNet,
    TensorNet,
    CHGNet,
    SO3Net,
    GRACE,
    QET,
    TransformedTargetModel,
    Potential,
    BondExpansion,
    RadialBesselFunction,
    FourierExpansion,
    ChebyshevRadialBasis,
    RealSphericalHarmonics,
    LinearQeq,
    get_available_pretrained_models,
)


@pytest.fixture
def synthetic_crystal():
    """Create a synthetic crystal graph for testing."""
    num_nodes = 4
    pos = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.5, 0.0],
        [0.5, 1.2, 0.8],
        [1.5, 1.5, 1.0],
    ], dtype=np.float32)
    edge_index = np.array([
        [0, 1, 1, 2, 2, 3, 3, 0],
        [1, 0, 2, 1, 3, 2, 0, 3],
    ], dtype=np.int32)
    line_edge_index = np.array([
        [0, 1, 2, 3],
        [1, 2, 3, 0],
    ], dtype=np.int32)
    node_type = np.array([6, 8, 1, 6], dtype=np.int32)
    batch = np.array([0, 0, 0, 0], dtype=np.int32)
    state_attr = np.array([[0.0, 0.0]], dtype=np.float32)
    return {
        "pos": pos,
        "edge_index": edge_index,
        "line_edge_index": line_edge_index,
        "node_type": node_type,
        "batch": batch,
        "state_attr": state_attr,
    }


def test_megnet_forward(synthetic_crystal):
    model = MEGNet(
        dim_node_embedding=16,
        dim_edge_embedding=20,
        dim_state_embedding=2,
        nblocks=2,
        hidden_layer_sizes_input=(32, 16),
        hidden_layer_sizes_conv=(32, 16),
        hidden_layer_sizes_output=(16,),
    )
    out = model(synthetic_crystal)
    assert out.shape == () or out.shape == (1,)
    assert np.isfinite(ops.convert_to_numpy(out)).all()


def test_m3gnet_forward(synthetic_crystal):
    model = M3GNet(
        dim_node_embedding=16,
        dim_edge_embedding=16,
        nblocks=2,
        units=16,
        max_n=3,
        max_l=3,
    )
    out = model(synthetic_crystal)
    assert out.shape == () or out.shape == (1,)
    assert np.isfinite(ops.convert_to_numpy(out)).all()


def test_tensornet_forward(synthetic_crystal):
    model = TensorNet(
        units=16,
        nblocks=2,
        num_rbf=16,
    )
    out = model(synthetic_crystal)
    assert out.shape == () or out.shape == (1,)
    assert np.isfinite(ops.convert_to_numpy(out)).all()


def test_chgnet_forward(synthetic_crystal):
    model = CHGNet(
        dim_atom_embedding=16,
        dim_bond_embedding=16,
        dim_angle_embedding=16,
        num_blocks=2,
        atom_conv_hidden_dims=(16,),
        bond_conv_hidden_dims=(16,),
    )
    out = model(synthetic_crystal)
    assert out.shape == () or out.shape == (1,)
    assert np.isfinite(ops.convert_to_numpy(out)).all()


def test_so3net_forward(synthetic_crystal):
    model = SO3Net(
        units=16,
        nblocks=2,
        lmax=2,
        num_rbf=16,
    )
    out = model(synthetic_crystal)
    assert out.shape == () or out.shape == (1,)
    assert np.isfinite(ops.convert_to_numpy(out)).all()


def test_grace_forward(synthetic_crystal):
    model = GRACE(
        cutoff=5.0,
        n_rad_base=6,
        lmax=2,
        embedding_size=8,
        max_order=2,
        nblocks=2,
        readout_hidden=(16,),
    )
    out = model(synthetic_crystal)
    assert out.shape == () or out.shape == (1,)
    assert np.isfinite(ops.convert_to_numpy(out)).all()


def test_qet_forward(synthetic_crystal):
    model = QET(
        units=16,
        nblocks=2,
        num_rbf=16,
    )
    out = model(synthetic_crystal)
    assert out.shape == () or out.shape == (1,)
    assert np.isfinite(ops.convert_to_numpy(out)).all()


def test_wrappers(synthetic_crystal):
    base_model = MEGNet(dim_node_embedding=8, dim_edge_embedding=16, nblocks=1)
    tt_model = TransformedTargetModel(model=base_model, mean=5.0, std=2.0)
    out_tt = tt_model(synthetic_crystal)
    assert np.isfinite(ops.convert_to_numpy(out_tt)).all()

    pot = Potential(model=base_model, data_mean=-1.5, data_std=0.8)
    out_pot = pot(synthetic_crystal)
    assert np.isfinite(ops.convert_to_numpy(out_pot)).all()


def test_available_models():
    models = get_available_pretrained_models()
    assert len(models) > 0
    assert any("MEGNet" in m for m in models)
    assert any("M3GNet" in m for m in models)

