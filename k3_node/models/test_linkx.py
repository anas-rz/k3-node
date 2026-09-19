import pytest
from keras import ops

from k3_node.models import LINKX


@pytest.mark.parametrize('num_edge_layers', [1, 2])
def test_linkx(num_edge_layers):
    x = ops.ones((4, 16))
    edge_index = ops.convert_to_tensor([[0, 1, 2], [1, 2, 3]])
    edge_weight = ops.convert_to_tensor([0.5, 0.8, 1.0])

    model = LINKX(
        num_nodes=4,
        in_channels=16,
        hidden_channels=32,
        out_channels=8,
        num_layers=2,
        num_edge_layers=num_edge_layers,
    )
    assert str(model) == 'LINKX(num_nodes=4, in_channels=16, out_channels=8)'

    out = model(x, edge_index)
    assert ops.shape(out) == (4, 8)

    out_no_x = model(None, edge_index)
    assert ops.shape(out_no_x) == (4, 8)

    out_weighted = model(x, edge_index, edge_weight)
    assert ops.shape(out_weighted) == (4, 8)

