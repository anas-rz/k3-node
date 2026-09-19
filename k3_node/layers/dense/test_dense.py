import math
import numpy as np
import pytest
from keras import layers, ops, random, Sequential

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


@pytest.mark.parametrize('weight', ['glorot', 'kaiming_uniform', None])
@pytest.mark.parametrize('bias', ['zeros', None])
def test_linear(weight, bias):
    x = random.normal((3, 4, 16))
    lin = Linear(16, 32, weight_initializer=weight, bias_initializer=bias)
    assert str(lin) == 'Linear(16, 32, bias=True)'
    out = lin(x)
    assert ops.shape(out) == (3, 4, 32)


@pytest.mark.parametrize('weight', ['glorot', 'kaiming_uniform', None])
@pytest.mark.parametrize('bias', ['zeros', None])
def test_lazy_linear(weight, bias):
    x = random.normal((3, 4, 16))
    lin = Linear(-1, 32, weight_initializer=weight, bias_initializer=bias)
    assert str(lin) == 'Linear(-1, 32, bias=True)'
    out = lin(x)
    assert ops.shape(out) == (3, 4, 32)
    assert str(lin) == 'Linear(16, 32, bias=True)'


def test_hetero_linear_basic():
    x = random.normal((3, 16))
    type_vec = ops.convert_to_tensor(np.array([0, 1, 2], dtype=np.int32))

    lin = HeteroLinear(16, 32, num_types=3)
    assert str(lin) == 'HeteroLinear(16, 32, num_types=3, bias=True)'

    out = lin(x, type_vec)
    assert ops.shape(out) == (3, 32)


def test_lazy_hetero_linear():
    x = random.normal((3, 16))
    type_vec = ops.convert_to_tensor(np.array([0, 1, 2], dtype=np.int32))

    lin = HeteroLinear(-1, 32, num_types=3)
    assert str(lin) == 'HeteroLinear(-1, 32, num_types=3, bias=True)'

    out = lin(x, type_vec)
    assert ops.shape(out) == (3, 32)


@pytest.mark.parametrize('bias', [True, False])
def test_hetero_dict_linear(bias):
    x_dict = {
        'v': random.normal((3, 16)),
        'w': random.normal((2, 8)),
    }

    lin = HeteroDictLinear({'v': 16, 'w': 8}, 32, bias=bias)
    assert str(lin) == f"HeteroDictLinear({{'v': 16, 'w': 8}}, 32, bias={bias})"

    out_dict = lin(x_dict)
    assert len(out_dict) == 2
    assert ops.shape(out_dict['v']) == (3, 32)
    assert ops.shape(out_dict['w']) == (2, 32)

    x_dict2 = {
        'v': random.normal((3, 16)),
        'w': random.normal((2, 16)),
    }

    lin2 = HeteroDictLinear(16, 32, types=['v', 'w'], bias=bias)
    assert str(lin2) == f"HeteroDictLinear({{'v': 16, 'w': 16}}, 32, bias={bias})"

    out_dict2 = lin2(x_dict2)
    assert len(out_dict2) == 2
    assert ops.shape(out_dict2['v']) == (3, 32)
    assert ops.shape(out_dict2['w']) == (2, 32)


def test_lazy_hetero_dict_linear():
    x_dict = {
        'v': random.normal((3, 16)),
        'w': random.normal((2, 8)),
    }

    lin = HeteroDictLinear(-1, 32, types=['v', 'w'])
    assert str(lin) == "HeteroDictLinear({'v': -1, 'w': -1}, 32, bias=True)"

    out_dict = lin(x_dict)
    assert len(out_dict) == 2
    assert ops.shape(out_dict['v']) == (3, 32)
    assert ops.shape(out_dict['w']) == (2, 32)


def test_dense_gcn_conv():
    channels = 16
    conv = DenseGCNConv(channels, channels)
    assert str(conv) == 'DenseGCNConv(16, 16)'

    x = random.normal((2, 3, channels))
    adj = ops.convert_to_tensor(np.array([
        [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
    ], dtype=np.float32))
    mask = ops.convert_to_tensor(np.array([[1, 1, 1], [1, 1, 0]], dtype=bool))

    out = conv(x, adj, mask)
    assert ops.shape(out) == (2, 3, channels)
    assert float(ops.sum(ops.abs(out[1, 2]))) == 0.0


def test_dense_gcn_conv_with_broadcasting():
    batch_size, num_nodes, channels = 8, 3, 16
    conv = DenseGCNConv(channels, channels)

    x = random.normal((batch_size, num_nodes, channels))
    adj = ops.convert_to_tensor(np.array([
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ], dtype=np.float32))

    assert ops.shape(conv(x, adj)) == (batch_size, num_nodes, channels)
    mask = ops.convert_to_tensor(np.array([1, 1, 1], dtype=bool))
    assert ops.shape(conv(x, adj, mask)) == (batch_size, num_nodes, channels)


def test_dense_gin_conv():
    channels = 16
    nn = Sequential([
        layers.Dense(channels, activation='relu'),
        layers.Dense(channels),
    ])
    dense_conv = DenseGINConv(nn)
    assert str(dense_conv) == f'DenseGINConv(nn={nn})'

    x = random.normal((2, 3, channels))
    adj = ops.convert_to_tensor(np.array([
        [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
    ], dtype=np.float32))
    mask = ops.convert_to_tensor(np.array([[1, 1, 1], [1, 1, 0]], dtype=bool))

    out = dense_conv(x, adj, mask)
    assert ops.shape(out) == (2, 3, channels)
    assert float(ops.sum(ops.abs(out[1, 2]))) == 0.0


def test_dense_gin_conv_with_broadcasting():
    batch_size, num_nodes, channels = 8, 3, 16
    nn = Sequential([
        layers.Dense(channels, activation='relu'),
        layers.Dense(channels),
    ])
    conv = DenseGINConv(nn)

    x = random.normal((batch_size, num_nodes, channels))
    adj = ops.convert_to_tensor(np.array([
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ], dtype=np.float32))

    assert ops.shape(conv(x, adj)) == (batch_size, num_nodes, channels)
    mask = ops.convert_to_tensor(np.array([1, 1, 1], dtype=bool))
    assert ops.shape(conv(x, adj, mask)) == (batch_size, num_nodes, channels)


@pytest.mark.parametrize('aggr', ['add', 'mean', 'max'])
def test_dense_graph_conv(aggr):
    channels = 16
    conv = DenseGraphConv(channels, channels, aggr=aggr)
    assert str(conv) == 'DenseGraphConv(16, 16)'

    x = random.normal((2, 3, channels))
    adj = ops.convert_to_tensor(np.array([
        [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
    ], dtype=np.float32))
    mask = ops.convert_to_tensor(np.array([[1, 1, 1], [1, 1, 0]], dtype=bool))

    out = conv(x, adj, mask)
    assert ops.shape(out) == (2, 3, channels)
    assert float(ops.sum(ops.abs(out[1, 2]))) == 0.0


@pytest.mark.parametrize('aggr', ['add', 'mean', 'max'])
def test_dense_graph_conv_with_broadcasting(aggr):
    batch_size, num_nodes, channels = 8, 3, 16
    conv = DenseGraphConv(channels, channels, aggr=aggr)

    x = random.normal((batch_size, num_nodes, channels))
    adj = ops.convert_to_tensor(np.array([
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ], dtype=np.float32))

    assert ops.shape(conv(x, adj)) == (batch_size, num_nodes, channels)
    mask = ops.convert_to_tensor(np.array([1, 1, 1], dtype=bool))
    assert ops.shape(conv(x, adj, mask)) == (batch_size, num_nodes, channels)


@pytest.mark.parametrize('normalize', [True, False])
def test_dense_sage_conv(normalize):
    channels = 16
    conv = DenseSAGEConv(channels, channels, normalize=normalize)
    assert str(conv) == 'DenseSAGEConv(16, 16)'

    x = random.normal((2, 3, channels))
    adj = ops.convert_to_tensor(np.array([
        [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
    ], dtype=np.float32))
    mask = ops.convert_to_tensor(np.array([[1, 1, 1], [1, 1, 0]], dtype=bool))

    out = conv(x, adj, mask)
    assert ops.shape(out) == (2, 3, channels)
    assert float(ops.sum(ops.abs(out[1, 2]))) == 0.0


def test_dense_sage_conv_with_broadcasting():
    batch_size, num_nodes, channels = 8, 3, 16
    conv = DenseSAGEConv(channels, channels)

    x = random.normal((batch_size, num_nodes, channels))
    adj = ops.convert_to_tensor(np.array([
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ], dtype=np.float32))

    assert ops.shape(conv(x, adj)) == (batch_size, num_nodes, channels)
    mask = ops.convert_to_tensor(np.array([1, 1, 1], dtype=bool))
    assert ops.shape(conv(x, adj, mask)) == (batch_size, num_nodes, channels)


@pytest.mark.parametrize('heads', [1, 4])
@pytest.mark.parametrize('concat', [True, False])
def test_dense_gat_conv(heads, concat):
    channels = 16
    conv = DenseGATConv(channels, channels, heads=heads, concat=concat)
    assert str(conv) == f'DenseGATConv(16, 16, heads={heads})'

    x = random.normal((2, 3, channels))
    adj = ops.convert_to_tensor(np.array([
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
    ], dtype=np.float32))
    mask = ops.convert_to_tensor(np.array([[1, 1, 1], [1, 1, 0]], dtype=bool))

    out = conv(x, adj, mask)
    out_dim = heads * channels if concat else channels
    assert ops.shape(out) == (2, 3, out_dim)
    assert float(ops.sum(ops.abs(out[1, 2]))) == 0.0


def test_dense_gat_conv_with_broadcasting():
    batch_size, num_nodes, channels = 8, 3, 16
    conv = DenseGATConv(channels, channels, heads=4)

    x = random.normal((batch_size, num_nodes, channels))
    adj = ops.convert_to_tensor(np.array([
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ], dtype=np.float32))

    assert ops.shape(conv(x, adj)) == (batch_size, num_nodes, 64)
    mask = ops.convert_to_tensor(np.array([1, 1, 1], dtype=bool))
    assert ops.shape(conv(x, adj, mask)) == (batch_size, num_nodes, 64)


def test_dense_diff_pool():
    batch_size, num_nodes, channels, num_clusters = (2, 20, 16, 10)
    x = random.normal((batch_size, num_nodes, channels))
    adj = ops.convert_to_tensor(np.random.rand(batch_size, num_nodes, num_nodes).astype(np.float32))
    s = random.normal((batch_size, num_nodes, num_clusters))
    mask = ops.convert_to_tensor(np.random.randint(0, 2, (batch_size, num_nodes), dtype=bool))

    x_out, adj_out, link_loss, ent_loss = dense_diff_pool(x, adj, s, mask)
    assert ops.shape(x_out) == (2, 10, 16)
    assert ops.shape(adj_out) == (2, 10, 10)
    assert float(link_loss) >= 0
    assert float(ent_loss) >= 0


def test_dense_mincut_pool():
    batch_size, num_nodes, channels, num_clusters = (2, 20, 16, 10)
    x = random.normal((batch_size, num_nodes, channels))
    adj = ops.ones((batch_size, num_nodes, num_nodes), dtype="float32")
    s = random.normal((batch_size, num_nodes, num_clusters))
    mask = ops.convert_to_tensor(np.random.randint(0, 2, (batch_size, num_nodes), dtype=bool))

    x_out, adj_out, mincut_loss, ortho_loss = dense_mincut_pool(x, adj, s, mask)
    assert ops.shape(x_out) == (2, 10, 16)
    assert ops.shape(adj_out) == (2, 10, 10)
    assert -1.0 <= float(mincut_loss) <= 0.0
    assert 0.0 <= float(ortho_loss) <= 2.0


def test_dmon_pooling():
    batch_size, num_nodes, channels, num_clusters = (2, 20, 16, 10)
    x = random.normal((batch_size, num_nodes, channels))
    adj = ops.ones((batch_size, num_nodes, num_nodes), dtype="float32")
    mask = ops.convert_to_tensor(np.random.randint(0, 2, (batch_size, num_nodes), dtype=bool))

    pool = DMoNPooling([channels, channels], num_clusters)
    assert str(pool) == 'DMoNPooling(16, num_clusters=10)'

    s, x_out, adj_out, spectral_loss, ortho_loss, cluster_loss = pool(x, adj, mask)
    assert ops.shape(s) == (2, 20, 10)
    assert ops.shape(x_out) == (2, 10, 16)
    assert ops.shape(adj_out) == (2, 10, 10)
    assert -1.0 <= float(spectral_loss) <= 0.5
    assert 0.0 <= float(ortho_loss) <= math.sqrt(2) + 1e-4
    assert 0.0 <= float(cluster_loss) <= math.sqrt(num_clusters) - 1 + 1e-4

