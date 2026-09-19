"""
Sanity-check reference parity tests against PyTorch Geometric.

These tests verify that k3-node dense layers produce numerically
equivalent outputs to torch_geometric.nn.dense reference implementations.
These tests are intended for local validation and documentation, and are
not run as part of the default GitHub Actions test suites.
"""

import numpy as np
import pytest
import torch
from keras import ops, layers, Sequential

# PyTorch Geometric references
from torch_geometric.nn.dense import (
    Linear as PyGLinear,
    HeteroLinear as PyGHeteroLinear,
    HeteroDictLinear as PyGHeteroDictLinear,
    DenseGCNConv as PyGDenseGCNConv,
    DenseGINConv as PyGDenseGINConv,
    DenseGraphConv as PyGDenseGraphConv,
    DenseSAGEConv as PyGDenseSAGEConv,
    DenseGATConv as PyGDenseGATConv,
    dense_diff_pool as pyg_diff_pool,
    dense_mincut_pool as pyg_mincut_pool,
    DMoNPooling as PyGDMoNPooling,
)

# k3-node implementations
from k3_node.layers.dense import (
    Linear,
    HeteroLinear,
    HeteroDictLinear,
    DenseGCNConv,
    DenseGINConv,
    DenseGraphConv,
    DenseSAGEConv,
    DenseGATConv,
    dense_diff_pool,
    dense_mincut_pool,
    DMoNPooling,
)


def test_reference_linear():
    torch.manual_seed(42)
    pyg_lin = PyGLinear(16, 32, bias=True)
    k3_lin = Linear(16, 32, bias=True)

    k3_lin.build((None, 16))
    with torch.no_grad():
        k3_lin.set_weights([
            pyg_lin.weight.detach().numpy().T,
            pyg_lin.bias.detach().numpy(),
        ])

    x_np = np.random.randn(3, 4, 16).astype(np.float32)
    pyg_out = pyg_lin(torch.from_numpy(x_np)).detach().numpy()
    k3_out = ops.convert_to_numpy(k3_lin(ops.convert_to_tensor(x_np)))

    assert np.allclose(pyg_out, k3_out, atol=1e-5)


def test_reference_hetero_linear():
    torch.manual_seed(42)
    pyg_lin = PyGHeteroLinear(16, 32, num_types=3, bias=True)
    k3_lin = HeteroLinear(16, 32, num_types=3, bias=True)

    k3_lin.build((None, 16))
    with torch.no_grad():
        # PyG weight shape: [num_types, in_channels, out_channels]
        # k3-node weight shape: [num_types, in_channels, out_channels]
        k3_lin.set_weights([
            pyg_lin.weight.detach().numpy(),
            pyg_lin.bias.detach().numpy(),
        ])

    x_np = np.random.randn(6, 16).astype(np.float32)
    type_vec_np = np.array([0, 1, 2, 0, 1, 2], dtype=np.int32)

    pyg_out = pyg_lin(torch.from_numpy(x_np), torch.from_numpy(type_vec_np).long()).detach().numpy()
    k3_out = ops.convert_to_numpy(k3_lin(ops.convert_to_tensor(x_np), ops.convert_to_tensor(type_vec_np)))

    assert np.allclose(pyg_out, k3_out, atol=1e-5)


def test_reference_hetero_dict_linear():
    torch.manual_seed(42)
    in_channels_dict = {'v': 16, 'w': 8}
    out_channels = 32
    pyg_lin = PyGHeteroDictLinear(in_channels_dict, out_channels, bias=True)
    k3_lin = HeteroDictLinear(in_channels_dict, out_channels, bias=True)

    for node_type, c_in in in_channels_dict.items():
        k3_lin.lins[node_type].build((None, c_in))
        with torch.no_grad():
            pyg_layer = pyg_lin.lins[node_type]
            k3_lin.lins[node_type].set_weights([
                pyg_layer.weight.detach().numpy().T,
                pyg_layer.bias.detach().numpy(),
            ])

    x_dict_np = {
        'v': np.random.randn(4, 16).astype(np.float32),
        'w': np.random.randn(3, 8).astype(np.float32),
    }

    pyg_input = {k: torch.from_numpy(v) for k, v in x_dict_np.items()}
    k3_input = {k: ops.convert_to_tensor(v) for k, v in x_dict_np.items()}

    pyg_out = pyg_lin(pyg_input)
    k3_out = k3_lin(k3_input)

    for k in in_channels_dict:
        assert np.allclose(pyg_out[k].detach().numpy(), ops.convert_to_numpy(k3_out[k]), atol=1e-5)


def test_reference_dense_gcn_conv():
    torch.manual_seed(42)
    channels = 16
    pyg_conv = PyGDenseGCNConv(channels, channels, bias=True)
    k3_conv = DenseGCNConv(channels, channels, bias=True)

    k3_conv.lin.build((None, channels))
    with torch.no_grad():
        k3_conv.lin.set_weights([
            pyg_conv.lin.weight.detach().numpy().T,
        ])
        k3_conv.bias.assign(ops.convert_to_tensor(pyg_conv.bias.detach().numpy()))

    x = np.random.randn(2, 4, channels).astype(np.float32)
    adj = np.random.rand(2, 4, 4).astype(np.float32)
    adj = adj + np.swapaxes(adj, 1, 2)
    mask = np.array([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=bool)

    pyg_out = pyg_conv(torch.from_numpy(x), torch.from_numpy(adj), torch.from_numpy(mask)).detach().numpy()
    k3_out = ops.convert_to_numpy(k3_conv(ops.convert_to_tensor(x), ops.convert_to_tensor(adj), ops.convert_to_tensor(mask)))

    assert np.allclose(pyg_out, k3_out, atol=1e-5)


def test_reference_dense_gin_conv():
    torch.manual_seed(42)
    channels = 16
    pyg_nn = torch.nn.Sequential(
        torch.nn.Linear(channels, channels),
        torch.nn.ReLU(),
        torch.nn.Linear(channels, channels),
    )
    k3_nn = Sequential([
        layers.Dense(channels, activation='relu'),
        layers.Dense(channels),
    ])

    pyg_conv = PyGDenseGINConv(pyg_nn, eps=0.5, train_eps=True)
    k3_conv = DenseGINConv(k3_nn, eps=0.5, train_eps=True)

    k3_nn.build((None, None, channels))
    with torch.no_grad():
        w0 = pyg_nn[0].weight.detach().numpy().T
        b0 = pyg_nn[0].bias.detach().numpy()
        w1 = pyg_nn[2].weight.detach().numpy().T
        b1 = pyg_nn[2].bias.detach().numpy()
        k3_nn.layers[0].set_weights([w0, b0])
        k3_nn.layers[1].set_weights([w1, b1])

    x = np.random.randn(2, 4, channels).astype(np.float32)
    adj = np.random.rand(2, 4, 4).astype(np.float32)
    adj = adj + np.swapaxes(adj, 1, 2)
    mask = np.array([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=bool)

    pyg_out = pyg_conv(torch.from_numpy(x), torch.from_numpy(adj), torch.from_numpy(mask)).detach().numpy()
    k3_out = ops.convert_to_numpy(k3_conv(ops.convert_to_tensor(x), ops.convert_to_tensor(adj), ops.convert_to_tensor(mask)))

    assert np.allclose(pyg_out, k3_out, atol=1e-5)


@pytest.mark.parametrize('aggr', ['add', 'mean', 'max'])
def test_reference_dense_graph_conv(aggr):
    torch.manual_seed(42)
    channels = 16
    pyg_conv = PyGDenseGraphConv(channels, channels, aggr=aggr, bias=True)
    k3_conv = DenseGraphConv(channels, channels, aggr=aggr, bias=True)

    k3_conv.build((None, None, channels))
    with torch.no_grad():
        k3_conv.lin_rel.set_weights([
            pyg_conv.lin_rel.weight.detach().numpy().T,
            pyg_conv.lin_rel.bias.detach().numpy(),
        ])
        k3_conv.lin_root.set_weights([
            pyg_conv.lin_root.weight.detach().numpy().T,
        ])

    x = np.random.randn(2, 4, channels).astype(np.float32)
    adj = np.random.rand(2, 4, 4).astype(np.float32)
    mask = np.array([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=bool)

    pyg_out = pyg_conv(torch.from_numpy(x), torch.from_numpy(adj), torch.from_numpy(mask)).detach().numpy()
    k3_out = ops.convert_to_numpy(k3_conv(ops.convert_to_tensor(x), ops.convert_to_tensor(adj), ops.convert_to_tensor(mask)))

    assert np.allclose(pyg_out, k3_out, atol=1e-5)


@pytest.mark.parametrize('normalize', [True, False])
def test_reference_dense_sage_conv(normalize):
    torch.manual_seed(42)
    channels = 16
    pyg_conv = PyGDenseSAGEConv(channels, channels, normalize=normalize, bias=True)
    k3_conv = DenseSAGEConv(channels, channels, normalize=normalize, bias=True)

    k3_conv.build((None, None, channels))
    with torch.no_grad():
        k3_conv.lin_rel.set_weights([
            pyg_conv.lin_rel.weight.detach().numpy().T,
        ])
        k3_conv.lin_root.set_weights([
            pyg_conv.lin_root.weight.detach().numpy().T,
            pyg_conv.lin_root.bias.detach().numpy(),
        ])

    x = np.random.randn(2, 4, channels).astype(np.float32)
    adj = np.random.rand(2, 4, 4).astype(np.float32)
    mask = np.array([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=bool)

    pyg_out = pyg_conv(torch.from_numpy(x), torch.from_numpy(adj), torch.from_numpy(mask)).detach().numpy()
    k3_out = ops.convert_to_numpy(k3_conv(ops.convert_to_tensor(x), ops.convert_to_tensor(adj), ops.convert_to_tensor(mask)))

    assert np.allclose(pyg_out, k3_out, atol=1e-5)


@pytest.mark.parametrize('heads', [1, 4])
@pytest.mark.parametrize('concat', [True, False])
def test_reference_dense_gat_conv(heads, concat):
    torch.manual_seed(42)
    channels = 16
    pyg_conv = PyGDenseGATConv(channels, channels, heads=heads, concat=concat, bias=True)
    k3_conv = DenseGATConv(channels, channels, heads=heads, concat=concat, bias=True)

    k3_conv.lin.build((None, channels))
    with torch.no_grad():
        w_lin = pyg_conv.lin.weight.detach().numpy().T
        k3_conv.lin.set_weights([w_lin])

        # att_src: [1, 1, heads, out_channels] -> [1, 1, heads, out_channels]
        # att_dst: [1, 1, heads, out_channels] -> [1, 1, heads, out_channels]
        k3_conv.att_src.assign(ops.convert_to_tensor(pyg_conv.att_src.detach().numpy()))
        k3_conv.att_dst.assign(ops.convert_to_tensor(pyg_conv.att_dst.detach().numpy()))
        k3_conv.bias.assign(ops.convert_to_tensor(pyg_conv.bias.detach().numpy()))

    x = np.random.randn(2, 4, channels).astype(np.float32)
    adj = np.random.rand(2, 4, 4).astype(np.float32)
    mask = np.array([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=bool)

    pyg_out = pyg_conv(torch.from_numpy(x), torch.from_numpy(adj), torch.from_numpy(mask)).detach().numpy()
    k3_out = ops.convert_to_numpy(k3_conv(ops.convert_to_tensor(x), ops.convert_to_tensor(adj), ops.convert_to_tensor(mask)))

    assert np.allclose(pyg_out, k3_out, atol=1e-4)


def test_reference_dense_diff_pool():
    np.random.seed(42)
    batch_size, num_nodes, channels, num_clusters = (2, 6, 8, 4)
    x = np.random.randn(batch_size, num_nodes, channels).astype(np.float32)
    adj = np.random.rand(batch_size, num_nodes, num_nodes).astype(np.float32)
    adj = adj + np.swapaxes(adj, 1, 2)
    s = np.random.randn(batch_size, num_nodes, num_clusters).astype(np.float32)
    mask = np.array([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 0, 0]], dtype=bool)

    pyg_res = pyg_diff_pool(
        torch.from_numpy(x),
        torch.from_numpy(adj),
        torch.from_numpy(s),
        torch.from_numpy(mask),
    )
    k3_res = dense_diff_pool(
        ops.convert_to_tensor(x),
        ops.convert_to_tensor(adj),
        ops.convert_to_tensor(s),
        ops.convert_to_tensor(mask),
    )

    for p, k in zip(pyg_res, k3_res):
        assert np.allclose(p.detach().numpy(), ops.convert_to_numpy(k), atol=1e-5)


def test_reference_dense_mincut_pool():
    np.random.seed(42)
    batch_size, num_nodes, channels, num_clusters = (2, 6, 8, 4)
    x = np.random.randn(batch_size, num_nodes, channels).astype(np.float32)
    adj = np.random.rand(batch_size, num_nodes, num_nodes).astype(np.float32)
    adj = adj + np.swapaxes(adj, 1, 2)
    s = np.random.randn(batch_size, num_nodes, num_clusters).astype(np.float32)
    mask = np.array([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 0, 0]], dtype=bool)

    pyg_res = pyg_mincut_pool(
        torch.from_numpy(x),
        torch.from_numpy(adj),
        torch.from_numpy(s),
        torch.from_numpy(mask),
    )
    k3_res = dense_mincut_pool(
        ops.convert_to_tensor(x),
        ops.convert_to_tensor(adj),
        ops.convert_to_tensor(s),
        ops.convert_to_tensor(mask),
    )

    for p, k in zip(pyg_res, k3_res):
        assert np.allclose(p.detach().numpy(), ops.convert_to_numpy(k), atol=1e-5)


def test_reference_dmon_pooling():
    torch.manual_seed(42)
    channels = 16
    num_clusters = 4
    pyg_pool = PyGDMoNPooling([channels, 8], num_clusters)
    k3_pool = DMoNPooling([channels, 8], num_clusters)

    k3_pool.mlp.lins[0].build((None, channels))
    k3_pool.mlp.lins[1].build((None, 8))
    with torch.no_grad():
        for i, lin in enumerate(pyg_pool.mlp.lins):
            k3_pool.mlp.lins[i].set_weights([
                lin.weight.detach().numpy().T,
                lin.bias.detach().numpy(),
            ])

    x = np.random.randn(2, 6, channels).astype(np.float32)
    adj = np.random.rand(2, 6, 6).astype(np.float32)
    adj = adj + np.swapaxes(adj, 1, 2)
    mask = np.array([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 0, 0]], dtype=bool)

    pyg_res = pyg_pool(torch.from_numpy(x), torch.from_numpy(adj), torch.from_numpy(mask))
    k3_res = k3_pool(ops.convert_to_tensor(x), ops.convert_to_tensor(adj), ops.convert_to_tensor(mask))

    for p, k in zip(pyg_res, k3_res):
        assert np.allclose(p.detach().numpy(), ops.convert_to_numpy(k), atol=1e-5)
