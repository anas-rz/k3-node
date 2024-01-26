import pytest
from keras import ops, random

from k3_node.layers import SAGEConv
from k3_node.utils import edge_index_to_adjacency_matrix


@pytest.mark.parametrize("out_channels", [32, 64])
@pytest.mark.parametrize("in_channels", [32, 64])
def test_graph_conv(in_channels, out_channels):
    x = random.normal((4, in_channels))
    edge_index = [[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]]
    adj1 = edge_index_to_adjacency_matrix(edge_index)

    conv = SAGEConv(out_channels)
    out = conv(x, adj1)
    assert ops.shape(out) == (4, out_channels)