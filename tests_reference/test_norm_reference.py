"""
Sanity-check reference parity tests against PyTorch Geometric.

These tests verify that k3-node normalization layers produce numerically
equivalent outputs to torch_geometric.nn.norm reference implementations.
These tests are intended for local validation and documentation, and are
not run as part of the default GitHub Actions test suites.
"""

import numpy as np
import pytest
import torch
from keras import ops

# PyTorch Geometric references
from torch_geometric.nn.norm import (
    BatchNorm as PyGBatchNorm,
    DiffGroupNorm as PyGDiffGroupNorm,
    GraphNorm as PyGGraphNorm,
    GraphSizeNorm as PyGGraphSizeNorm,
    HeteroBatchNorm as PyGHeteroBatchNorm,
    HeteroLayerNorm as PyGHeteroLayerNorm,
    InstanceNorm as PyGInstanceNorm,
    LayerNorm as PyGLayerNorm,
    MeanSubtractionNorm as PyGMeanSubtractionNorm,
    MessageNorm as PyGMessageNorm,
    PairNorm as PyGPairNorm,
)

# k3-node implementations
from k3_node.layers.norm import (
    BatchNorm,
    DiffGroupNorm,
    GraphNorm,
    GraphSizeNorm,
    HeteroBatchNorm,
    HeteroLayerNorm,
    InstanceNorm,
    LayerNorm,
    MeanSubtractionNorm,
    MessageNorm,
    PairNorm,
)


def test_reference_graph_size_norm():
    np.random.seed(42)
    x_np = np.random.randn(100, 16).astype(np.float32)
    batch_np = np.repeat(np.arange(10), 10).astype(np.int32)

    # Without batch
    pyg_norm = PyGGraphSizeNorm()
    k3_norm = GraphSizeNorm()
    pyg_out = pyg_norm(torch.from_numpy(x_np)).detach().numpy()
    k3_out = ops.convert_to_numpy(k3_norm(ops.convert_to_tensor(x_np)))
    assert np.allclose(pyg_out, k3_out, atol=1e-5)

    # With batch
    pyg_out_b = pyg_norm(torch.from_numpy(x_np), torch.from_numpy(batch_np).long()).detach().numpy()
    k3_out_b = ops.convert_to_numpy(k3_norm(ops.convert_to_tensor(x_np), ops.convert_to_tensor(batch_np)))
    assert np.allclose(pyg_out_b, k3_out_b, atol=1e-5)


@pytest.mark.parametrize("scale_individually", [False, True])
def test_reference_pair_norm(scale_individually):
    np.random.seed(42)
    x_np = np.random.randn(100, 16).astype(np.float32)
    batch_np = np.repeat(np.arange(4), 25).astype(np.int32)

    pyg_norm = PyGPairNorm(scale_individually=scale_individually)
    k3_norm = PairNorm(scale_individually=scale_individually)

    # Without batch
    pyg_out = pyg_norm(torch.from_numpy(x_np)).detach().numpy()
    k3_out = ops.convert_to_numpy(k3_norm(ops.convert_to_tensor(x_np)))
    assert np.allclose(pyg_out, k3_out, atol=1e-5)

    # With batch
    pyg_out_b = pyg_norm(torch.from_numpy(x_np), torch.from_numpy(batch_np).long()).detach().numpy()
    k3_out_b = ops.convert_to_numpy(k3_norm(ops.convert_to_tensor(x_np), ops.convert_to_tensor(batch_np)))
    assert np.allclose(pyg_out_b, k3_out_b, atol=1e-5)


def test_reference_mean_subtraction_norm():
    np.random.seed(42)
    x_np = np.random.randn(6, 16).astype(np.float32)
    batch_np = np.array([0, 0, 1, 1, 1, 2], dtype=np.int32)

    pyg_norm = PyGMeanSubtractionNorm()
    k3_norm = MeanSubtractionNorm()

    # Without batch
    pyg_out = pyg_norm(torch.from_numpy(x_np)).detach().numpy()
    k3_out = ops.convert_to_numpy(k3_norm(ops.convert_to_tensor(x_np)))
    assert np.allclose(pyg_out, k3_out, atol=1e-5)

    # With batch
    pyg_out_b = pyg_norm(torch.from_numpy(x_np), torch.from_numpy(batch_np).long()).detach().numpy()
    k3_out_b = ops.convert_to_numpy(k3_norm(ops.convert_to_tensor(x_np), ops.convert_to_tensor(batch_np)))
    assert np.allclose(pyg_out_b, k3_out_b, atol=1e-5)


@pytest.mark.parametrize("learn_scale", [False, True])
def test_reference_message_norm(learn_scale):
    np.random.seed(42)
    x_np = np.random.randn(100, 16).astype(np.float32)
    msg_np = np.random.randn(100, 16).astype(np.float32)

    pyg_norm = PyGMessageNorm(learn_scale=learn_scale)
    k3_norm = MessageNorm(learn_scale=learn_scale)

    pyg_out = pyg_norm(torch.from_numpy(x_np), torch.from_numpy(msg_np)).detach().numpy()
    k3_out = ops.convert_to_numpy(k3_norm(ops.convert_to_tensor(x_np), ops.convert_to_tensor(msg_np)))
    assert np.allclose(pyg_out, k3_out, atol=1e-5)


def test_reference_graph_norm():
    np.random.seed(42)
    x_np = np.random.randn(200, 16).astype(np.float32)
    batch_np = np.repeat(np.arange(4), 50).astype(np.int32)

    pyg_norm = PyGGraphNorm(16)
    k3_norm = GraphNorm(16)

    # Without batch
    pyg_out = pyg_norm(torch.from_numpy(x_np)).detach().numpy()
    k3_out = ops.convert_to_numpy(k3_norm(ops.convert_to_tensor(x_np)))
    assert np.allclose(pyg_out, k3_out, atol=1e-5)

    # With batch
    pyg_out_b = pyg_norm(torch.from_numpy(x_np), torch.from_numpy(batch_np).long()).detach().numpy()
    k3_out_b = ops.convert_to_numpy(k3_norm(ops.convert_to_tensor(x_np), ops.convert_to_tensor(batch_np)))
    assert np.allclose(pyg_out_b, k3_out_b, atol=1e-5)


@pytest.mark.parametrize("affine", [False, True])
def test_reference_instance_norm(affine):
    np.random.seed(42)
    x_np = np.random.randn(100, 16).astype(np.float32)
    batch_np = np.repeat(np.arange(4), 25).astype(np.int32)

    pyg_norm = PyGInstanceNorm(16, affine=affine, track_running_stats=False)
    k3_norm = InstanceNorm(16, affine=affine, track_running_stats=False)

    # Without batch
    pyg_out = pyg_norm(torch.from_numpy(x_np)).detach().numpy()
    k3_out = ops.convert_to_numpy(k3_norm(ops.convert_to_tensor(x_np)))
    assert np.allclose(pyg_out, k3_out, atol=1e-5)

    # With batch
    pyg_out_b = pyg_norm(torch.from_numpy(x_np), torch.from_numpy(batch_np).long()).detach().numpy()
    k3_out_b = ops.convert_to_numpy(k3_norm(ops.convert_to_tensor(x_np), ops.convert_to_tensor(batch_np)))
    assert np.allclose(pyg_out_b, k3_out_b, atol=1e-5)


@pytest.mark.parametrize("mode", ["graph", "node"])
@pytest.mark.parametrize("affine", [False, True])
def test_reference_layer_norm(mode, affine):
    np.random.seed(42)
    x_np = np.random.randn(100, 16).astype(np.float32)
    batch_np = np.repeat(np.arange(4), 25).astype(np.int32)

    pyg_norm = PyGLayerNorm(16, affine=affine, mode=mode)
    k3_norm = LayerNorm(16, affine=affine, mode=mode)

    # Without batch
    pyg_out = pyg_norm(torch.from_numpy(x_np)).detach().numpy()
    k3_out = ops.convert_to_numpy(k3_norm(ops.convert_to_tensor(x_np)))
    assert np.allclose(pyg_out, k3_out, atol=1e-5)

    # With batch
    pyg_out_b = pyg_norm(torch.from_numpy(x_np), torch.from_numpy(batch_np).long()).detach().numpy()
    k3_out_b = ops.convert_to_numpy(k3_norm(ops.convert_to_tensor(x_np), ops.convert_to_tensor(batch_np)))
    assert np.allclose(pyg_out_b, k3_out_b, atol=1e-5)


@pytest.mark.parametrize("affine", [False, True])
def test_reference_hetero_layer_norm(affine):
    np.random.seed(42)
    x_np = np.random.randn(100, 16).astype(np.float32)
    type_vec_np = np.repeat(np.arange(4), 25).astype(np.int32)
    type_ptr = [0, 25, 50, 75, 100]

    pyg_norm = PyGHeteroLayerNorm(16, num_types=4, affine=affine)
    k3_norm = HeteroLayerNorm(16, num_types=4, affine=affine)

    pyg_out = pyg_norm(torch.from_numpy(x_np), torch.from_numpy(type_vec_np).long()).detach().numpy()
    k3_out = ops.convert_to_numpy(k3_norm(ops.convert_to_tensor(x_np), ops.convert_to_tensor(type_vec_np)))
    assert np.allclose(pyg_out, k3_out, atol=1e-4)

    # With type_ptr
    pyg_out_ptr = pyg_norm(torch.from_numpy(x_np), type_ptr=type_ptr).detach().numpy()
    k3_out_ptr = ops.convert_to_numpy(k3_norm(ops.convert_to_tensor(x_np), type_ptr=type_ptr))
    assert np.allclose(pyg_out_ptr, k3_out_ptr, atol=1e-4)


@pytest.mark.parametrize("affine", [False, True])
def test_reference_batch_norm(affine):
    np.random.seed(42)
    x_np = np.random.randn(100, 16).astype(np.float32)

    pyg_norm = PyGBatchNorm(16, affine=affine)
    k3_norm = BatchNorm(16, affine=affine)

    pyg_out = pyg_norm(torch.from_numpy(x_np)).detach().numpy()
    k3_out = ops.convert_to_numpy(k3_norm(ops.convert_to_tensor(x_np)))
    assert np.allclose(pyg_out, k3_out, atol=1e-5)


@pytest.mark.parametrize("affine", [False, True])
def test_reference_hetero_batch_norm(affine):
    np.random.seed(42)
    x_np = np.random.randn(100, 16).astype(np.float32)
    type_vec_np = np.repeat(np.arange(4), 25).astype(np.int32)

    pyg_norm = PyGHeteroBatchNorm(16, num_types=4, affine=affine)
    k3_norm = HeteroBatchNorm(16, num_types=4, affine=affine)

    pyg_out = pyg_norm(torch.from_numpy(x_np), torch.from_numpy(type_vec_np).long()).detach().numpy()
    k3_out = ops.convert_to_numpy(k3_norm(ops.convert_to_tensor(x_np), ops.convert_to_tensor(type_vec_np)))
    assert np.allclose(pyg_out, k3_out, atol=1e-4)


def test_reference_diff_group_norm_group_distance_ratio():
    np.random.seed(42)
    x_np = np.random.randn(6, 16).astype(np.float32)
    y_np = np.array([0, 1, 0, 1, 1, 1], dtype=np.int64)

    pyg_ratio = PyGDiffGroupNorm.group_distance_ratio(torch.from_numpy(x_np), torch.from_numpy(y_np))
    k3_ratio = DiffGroupNorm.group_distance_ratio(ops.convert_to_tensor(x_np), ops.convert_to_tensor(y_np))

    assert np.isclose(pyg_ratio, k3_ratio, atol=1e-5)

