"""
Sanity-check reference parity tests against PyTorch Geometric.

These tests verify that k3-node unpooling operators produce numerically
equivalent outputs to torch_geometric.nn.unpool reference implementations.
These tests are intended for local validation and documentation, and are
not run as part of the default GitHub Actions test suites.

Note: torch_geometric.nn.pool.knn (and therefore
torch_geometric.nn.unpool.knn_interpolate) hard-requires `pyg-lib`, which is
not installable in this environment. `torch-cluster` provides a numerically
equivalent k-NN search, so the PyG reference below reimplements
`knn_interpolate` verbatim from its PyG source, swapping only the k-NN
backend from `pyg-lib` to `torch-cluster`.
"""

import os
os.environ["KERAS_BACKEND"] = "torch"
import numpy as np
import torch
from keras import ops
from torch_cluster import knn as pyg_knn_ext
from torch_geometric.utils import scatter as pyg_scatter

from k3_node.layers.unpool import knn_interpolate


def _pyg_knn_interpolate(x, pos_x, pos_y, batch_x, batch_y, k=3):
    with torch.no_grad():
        assign_index = pyg_knn_ext(pos_x, pos_y, k, batch_x, batch_y)
        y_idx, x_idx = assign_index[0], assign_index[1]
        diff = pos_x[x_idx] - pos_y[y_idx]
        squared_distance = (diff * diff).sum(dim=-1, keepdim=True)
        weights = 1.0 / torch.clamp(squared_distance, min=1e-16)

    y = pyg_scatter(x[x_idx] * weights, y_idx, 0, pos_y.size(0), reduce="sum")
    y = y / pyg_scatter(weights, y_idx, 0, pos_y.size(0), reduce="sum")
    return y


def test_reference_knn_interpolate_fixed_case():
    # Matches torch_geometric/test/nn/unpool/test_knn_interpolate.py exactly.
    x = torch.tensor([[1.0], [10.0], [100.0], [-1.0], [-10.0], [-100.0]])
    pos_x = torch.tensor([
        [-1.0, 0.0], [0.0, 0.0], [1.0, 0.0],
        [-2.0, 0.0], [0.0, 0.0], [2.0, 0.0],
    ])
    pos_y = torch.tensor([
        [-1.0, -1.0], [1.0, 1.0], [-2.0, -2.0], [2.0, 2.0],
    ])
    batch_x = torch.tensor([0, 0, 0, 1, 1, 1])
    batch_y = torch.tensor([0, 0, 1, 1])

    out_pyg = _pyg_knn_interpolate(x, pos_x, pos_y, batch_x, batch_y, k=2)
    out_k3 = knn_interpolate(x, pos_x, pos_y, batch_x, batch_y, k=2)

    assert np.allclose(out_pyg.numpy(), ops.convert_to_numpy(out_k3), atol=1e-5)
    assert out_pyg.tolist() == [[4.0], [70.0], [-4.0], [-70.0]]


def test_reference_knn_interpolate_random():
    torch.manual_seed(42)
    x = torch.randn(20, 8)
    pos_x = torch.randn(20, 3)
    pos_y = torch.randn(12, 3)
    batch_x = torch.tensor([0] * 10 + [1] * 10)
    batch_y = torch.tensor([0] * 6 + [1] * 6)

    out_pyg = _pyg_knn_interpolate(x, pos_x, pos_y, batch_x, batch_y, k=4)
    out_k3 = knn_interpolate(x, pos_x, pos_y, batch_x, batch_y, k=4)

    assert np.allclose(out_pyg.numpy(), ops.convert_to_numpy(out_k3), atol=1e-5)
