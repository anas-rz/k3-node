"""
Sanity-check reference parity tests against PyTorch Geometric.

These tests verify that k3-node functional operators produce numerically
equivalent outputs to torch_geometric.nn.functional reference
implementations. These tests are intended for local validation and
documentation, and are not run as part of the default GitHub Actions test
suites.
"""

import os
os.environ["KERAS_BACKEND"] = "torch"
import numpy as np
import torch
from keras import ops

from torch_geometric.nn.functional import bro as pyg_bro, gini as pyg_gini
from k3_node.layers.functional import bro, gini


def test_reference_bro():
    np.random.seed(0)
    batch = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2], dtype=np.int64)
    x = np.random.randn(9, 4).astype(np.float32)

    out_pyg = pyg_bro(torch.from_numpy(x), torch.from_numpy(batch))
    out_k3 = bro(ops.convert_to_tensor(x), ops.convert_to_tensor(batch))

    assert np.allclose(out_pyg.item(), ops.convert_to_numpy(out_k3), atol=1e-4)


def test_reference_bro_fixed_case():
    # Matches torch_geometric/test/nn/functional/test_bro.py.
    batch = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2], dtype=np.int64)

    g1 = np.array([
        [0.2, 0.2, 0.2, 0.2],
        [0.0, 0.2, 0.2, 0.2],
        [0.2, 0.0, 0.2, 0.2],
        [0.2, 0.2, 0.0, 0.2],
    ], dtype=np.float32)
    g2 = np.array([
        [0.2, 0.2, 0.2, 0.2],
        [0.0, 0.2, 0.2, 0.2],
        [0.2, 0.0, 0.2, 0.2],
    ], dtype=np.float32)
    g3 = np.array([
        [0.2, 0.2, 0.2, 0.2],
        [0.2, 0.0, 0.2, 0.2],
    ], dtype=np.float32)
    x = np.concatenate([g1, g2, g3], axis=0)

    out_pyg = pyg_bro(torch.from_numpy(x), torch.from_numpy(batch))
    out_k3 = bro(ops.convert_to_tensor(x), ops.convert_to_tensor(batch))

    expected = sum(np.linalg.norm(g @ g.T - np.eye(g.shape[0])) for g in [g1, g2, g3]) / 3.0
    assert np.isclose(out_pyg.item(), expected, atol=1e-5)
    assert np.allclose(ops.convert_to_numpy(out_k3), expected, atol=1e-4)


def test_reference_gini():
    w = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1000.0]], dtype=np.float32)

    out_pyg = pyg_gini(torch.from_numpy(w))
    out_k3 = gini(ops.convert_to_tensor(w))

    assert np.isclose(out_pyg.item(), 0.5, atol=1e-5)
    assert np.allclose(ops.convert_to_numpy(out_k3), out_pyg.item(), atol=1e-5)


def test_reference_gini_random():
    np.random.seed(1)
    w = np.abs(np.random.randn(5, 6)).astype(np.float32)

    out_pyg = pyg_gini(torch.from_numpy(w))
    out_k3 = gini(ops.convert_to_tensor(w))

    assert np.allclose(ops.convert_to_numpy(out_k3), out_pyg.item(), atol=1e-4)
