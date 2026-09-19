import pytest
from keras import ops

from k3_node.models import GCN, GraphSAGE, GIN, GAT, PNA, EdgeCNN


@pytest.mark.parametrize('jk', [None, 'last', 'cat', 'max', 'lstm'])
@pytest.mark.parametrize('out_dim', [None, 4])
def test_gcn(jk, out_dim):
    x = ops.ones((4, 8))
    edge_index = ops.convert_to_tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    edge_weight = ops.convert_to_tensor([1.0, 1.0, 0.5, 0.5])
    out_channels = 16 if out_dim is None else out_dim

    model = GCN(8, 16, num_layers=3, out_channels=out_dim, dropout=0.1,
                act='relu', norm='batch_norm', jk=jk)
    assert str(model) == f'GCN(8, {out_channels}, num_layers=3)'

    out = model(x, edge_index, edge_weight=edge_weight)
    assert ops.shape(out) == (4, out_channels)


@pytest.mark.parametrize('jk', [None, 'cat', 'max'])
def test_graph_sage(jk):
    x = ops.ones((4, 8))
    edge_index = ops.convert_to_tensor([[0, 1, 1, 2], [1, 0, 2, 1]])

    model = GraphSAGE(8, 16, num_layers=2, out_channels=8, jk=jk)
    assert str(model) == 'GraphSAGE(8, 8, num_layers=2)'

    out = model(x, edge_index)
    assert ops.shape(out) == (4, 8)


@pytest.mark.parametrize('jk', [None, 'cat'])
def test_gin(jk):
    x = ops.ones((4, 8))
    edge_index = ops.convert_to_tensor([[0, 1, 1, 2], [1, 0, 2, 1]])

    model = GIN(8, 16, num_layers=2, out_channels=4, jk=jk)
    assert str(model) == 'GIN(8, 4, num_layers=2)'

    out = model(x, edge_index)
    assert ops.shape(out) == (4, 4)


@pytest.mark.parametrize('v2', [False, True])
def test_gat(v2):
    x = ops.ones((4, 8))
    edge_index = ops.convert_to_tensor([[0, 1, 1, 2], [1, 0, 2, 1]])

    model = GAT(8, 16, num_layers=2, out_channels=8, heads=2, v2=v2)
    assert str(model) == 'GAT(8, 8, num_layers=2)'

    out = model(x, edge_index)
    assert ops.shape(out) == (4, 8)


def test_pna():
    x = ops.ones((4, 8))
    edge_index = ops.convert_to_tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    deg = ops.convert_to_tensor([0, 1, 2, 1])

    model = PNA(
        8, 16, num_layers=2, out_channels=4,
        aggregators=['mean', 'min', 'max', 'std'],
        scalers=['identity', 'amplification', 'attenuation'],
        deg=deg,
    )
    assert str(model) == 'PNA(8, 4, num_layers=2)'

    out = model(x, edge_index)
    assert ops.shape(out) == (4, 4)


def test_edge_cnn():
    x = ops.ones((4, 8))
    edge_index = ops.convert_to_tensor([[0, 1, 1, 2], [1, 0, 2, 1]])

    model = EdgeCNN(8, 16, num_layers=2, out_channels=4)
    assert str(model) == 'EdgeCNN(8, 4, num_layers=2)'

    out = model(x, edge_index)
    assert ops.shape(out) == (4, 4)

