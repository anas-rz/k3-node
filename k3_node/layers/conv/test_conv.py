import pytest
from keras import ops, random, backend

from k3_node.utils.backend_import import *
from k3_node.layers.conv import (
    AGNNConv,
    APPNPConv,
    ARMAConv,
    CrystalConv,
    GatedGraphConv,
    GraphConvolution,
    GeneralConv,
    GINConv,
    GraphAttention,
    PPNPPropagation,
    SAGEConv,
    DiffusionConv
)
from k3_node.utils import edge_index_to_adjacency_matrix


@pytest.mark.parametrize("trainable", [True, False])
@pytest.mark.parametrize('sparse', [True, False])
def test_agnn_conv(trainable, sparse):
    if sparse and backend.backend() != 'tensorflow':
        pytest.skip('JAX and PyTorch backends donot support Sparse Adjacency matrices')
    x = random.normal((4, 16))
    edge_index = [[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]]
    adj1 = edge_index_to_adjacency_matrix(edge_index)
    if sparse:
        adj1 = tf.sparse.from_dense(adj1)
    conv = AGNNConv(trainable=trainable)
    out = conv(x, adj1)
    assert ops.shape(out) == (4, 16)


@pytest.mark.parametrize("channels", [32, 64])
@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.parametrize('sparse', [True, False])
def test_appnp_conv(channels, use_bias, sparse):
    if sparse and backend.backend() != 'tensorflow':
        pytest.skip('JAX and PyTorch backends donot support Sparse Adjacency matrices')
    x = random.normal((4, 16))
    edge_index = [[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]]
    adj1 = edge_index_to_adjacency_matrix(edge_index)
    if sparse:
        adj1 = tf.sparse.from_dense(adj1)
    conv = APPNPConv(
        channels,
        use_bias=use_bias,
    )
    out = conv((x, adj1))
    assert ops.shape(out) == (4, channels)


@pytest.mark.parametrize("channels", [32, 64])
@pytest.mark.parametrize("iterations", [1, 2, 3, 4])
@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.parametrize("sparse", [True, False])
def test_arma_conv(channels, iterations, use_bias, sparse):
    if sparse and backend.backend() != 'tensorflow':
        pytest.skip('JAX and PyTorch backends donot support Sparse Adjacency matrices')
    x = random.normal((4, 16))
    edge_index = [[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]]
    adj1 = edge_index_to_adjacency_matrix(edge_index)
    if sparse:
        adj1 = tf.sparse.from_dense(adj1)
    conv = ARMAConv(
        channels,
        iterations=iterations,
        use_bias=use_bias,
    )
    out = conv((x, adj1))
    assert ops.shape(out) == (4, channels)

@pytest.mark.parametrize("out_channels", [32, 64])
@pytest.mark.parametrize("K", [6, 8])
@pytest.mark.parametrize("in_channels", [32, 64])
@pytest.mark.parametrize("sparse", [False])
def test_diffusion_conv(out_channels, in_channels, K, sparse):
    if sparse and backend.backend() != 'tensorflow':
        pytest.skip('JAX and PyTorch backends donot support Sparse Adjacency matrices')
    if backend.backend() == 'torch':
        pytest.skip("DiffusionConv Doesn't work with PyTorch backend. Support will be added in future.")
    x = random.normal((4, in_channels))
    edge_index = [[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]]
    adj1 = edge_index_to_adjacency_matrix(edge_index)
    if sparse:
        adj1 = tf.sparse.from_dense(adj1)
    conv = DiffusionConv(out_channels,
        K=K)
    out = conv((x, adj1))
    assert ops.shape(out) == (4, out_channels)


@pytest.mark.parametrize("aggregate", ["sum", "max"])
@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.parametrize("in_channels", [32, 64])
@pytest.mark.parametrize("sparse", [True, False])
def test_crystal_conv(in_channels, aggregate, use_bias, sparse):
    if sparse and backend.backend() != 'tensorflow':
        pytest.skip('JAX and PyTorch backends donot support Sparse Adjacency matrices')
    x = random.normal((4, in_channels))
    edge_index = [[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]]
    adj1 = edge_index_to_adjacency_matrix(edge_index)
    if sparse:
        adj1 = tf.sparse.from_dense(adj1)
    conv = CrystalConv(aggregate=aggregate, use_bias=use_bias)
    out = conv((x, adj1))
    assert ops.shape(out) == (4, in_channels)


@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.parametrize("in_channels", [32])
@pytest.mark.parametrize("out_channels", [32, 64])
@pytest.mark.parametrize("n_layers", [1, 2])
@pytest.mark.parametrize("sparse", [True, False])
def test_gated_conv(in_channels, out_channels, n_layers, use_bias, sparse):
    if sparse and backend.backend() != 'tensorflow':
        pytest.skip('JAX and PyTorch backends donot support Sparse Adjacency matrices')
    x = random.normal((4, in_channels))
    edge_index = [[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]]
    adj1 = edge_index_to_adjacency_matrix(edge_index)
    if sparse:
        adj1 = tf.sparse.from_dense(adj1)
    conv = GatedGraphConv(out_channels, n_layers, use_bias=use_bias)
    out = conv((x, adj1))
    assert ops.shape(out) == (4, out_channels)


### TEST GCN


@pytest.mark.parametrize("channels", [32, 64])
@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.parametrize("sparse", [True, False])
def test_general_conv(channels, use_bias, sparse):
    if sparse and backend.backend() != 'tensorflow':
        pytest.skip('JAX and PyTorch backends donot support Sparse Adjacency matrices')
    x = random.normal((4, 16))
    edge_index = [[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]]
    adj1 = edge_index_to_adjacency_matrix(edge_index)
    if sparse:
        adj1 = tf.sparse.from_dense(adj1)
    conv = GeneralConv(
        channels,
        use_bias=use_bias,
    )
    out = conv((x, adj1))
    assert ops.shape(out) == (4, channels)


@pytest.mark.parametrize("out_channels", [32, 64])
@pytest.mark.parametrize("in_channels", [32, 64])
@pytest.mark.parametrize("sparse", [False])
def test_sage_conv(in_channels, out_channels, sparse):
    x = random.normal((4, in_channels))
    edge_index = [[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]]
    adj1 = edge_index_to_adjacency_matrix(edge_index)
    if sparse:
        adj1 = tf.sparse.from_dense(adj1)
    conv = SAGEConv(out_channels)
    out = conv(x, adj1)
    assert ops.shape(out) == (4, out_channels)
