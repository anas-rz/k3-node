"""
Training verification tests for all MatGL materials models.

Verifies that models can compute losses, calculate gradients, execute
training steps, and that training successfully reduces loss.
"""

from __future__ import annotations

import numpy as np
import pytest
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
)


@pytest.fixture
def synthetic_crystal():
    """Create a synthetic crystal graph for training tests."""
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


def test_megnet_training(synthetic_crystal):
    model = MEGNet(
        dim_node_embedding=16,
        dim_edge_embedding=20,
        dim_state_embedding=2,
        nblocks=2,
        hidden_layer_sizes_input=(32, 16),
        hidden_layer_sizes_conv=(32, 16),
        hidden_layer_sizes_output=(16,),
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss="mse")
    target = np.array([1.5], dtype=np.float32)

    l0 = float(model.train_on_batch(synthetic_crystal, target))
    for _ in range(8):
        l_last = float(model.train_on_batch(synthetic_crystal, target))

    assert l_last < l0, f"MEGNet loss did not decrease: initial={l0:.4f}, final={l_last:.4f}"


def test_m3gnet_training(synthetic_crystal):
    model = M3GNet(
        dim_node_embedding=16,
        dim_edge_embedding=16,
        nblocks=1,
        units=16,
        max_n=3,
        max_l=3,
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss="mse")
    target = np.array([2.0], dtype=np.float32)

    l0 = float(model.train_on_batch(synthetic_crystal, target))
    for _ in range(8):
        l_last = float(model.train_on_batch(synthetic_crystal, target))

    assert l_last < l0, f"M3GNet loss did not decrease: initial={l0:.4f}, final={l_last:.4f}"


def test_tensornet_training(synthetic_crystal):
    model = TensorNet(
        units=16,
        nblocks=1,
        num_rbf=8,
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss="mse")
    target = np.array([2.0], dtype=np.float32)

    l0 = float(model.train_on_batch(synthetic_crystal, target))
    for _ in range(8):
        l_last = float(model.train_on_batch(synthetic_crystal, target))

    assert l_last < l0, f"TensorNet loss did not decrease: initial={l0:.4f}, final={l_last:.4f}"


def test_chgnet_training(synthetic_crystal):
    model = CHGNet(
        dim_atom_embedding=16,
        dim_bond_embedding=16,
        dim_angle_embedding=16,
        num_blocks=1,
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.005), loss="mse")
    target = np.array([5.0], dtype=np.float32)

    l0 = float(model.train_on_batch(synthetic_crystal, target))
    for _ in range(10):
        l_last = float(model.train_on_batch(synthetic_crystal, target))

    assert l_last < l0, f"CHGNet loss did not decrease: initial={l0:.4f}, final={l_last:.4f}"


def test_so3net_training(synthetic_crystal):
    model = SO3Net(
        units=16,
        nblocks=1,
        lmax=2,
        num_rbf=8,
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.005), loss="mse")
    target = np.array([5.0], dtype=np.float32)

    l0 = float(model.train_on_batch(synthetic_crystal, target))
    for _ in range(8):
        l_last = float(model.train_on_batch(synthetic_crystal, target))

    assert l_last < l0, f"SO3Net loss did not decrease: initial={l0:.4f}, final={l_last:.4f}"


def test_grace_training(synthetic_crystal):
    model = GRACE(
        cutoff=5.0,
        n_rad_base=4,
        lmax=2,
        embedding_size=8,
        max_order=2,
        nblocks=1,
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.005), loss="mse")
    target = np.array([5.0], dtype=np.float32)

    l0 = float(model.train_on_batch(synthetic_crystal, target))
    for _ in range(8):
        l_last = float(model.train_on_batch(synthetic_crystal, target))

    assert l_last < l0, f"GRACE loss did not decrease: initial={l0:.4f}, final={l_last:.4f}"


def test_qet_training(synthetic_crystal):
    model = QET(
        units=16,
        nblocks=1,
        num_rbf=8,
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.005), loss="mse")
    target = np.array([5.0], dtype=np.float32)

    l0 = float(model.train_on_batch(synthetic_crystal, target))
    for _ in range(8):
        l_last = float(model.train_on_batch(synthetic_crystal, target))

    assert l_last < l0, f"QET loss did not decrease: initial={l0:.4f}, final={l_last:.4f}"

