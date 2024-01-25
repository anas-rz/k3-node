import pytest
from keras import ops, random

from k3_node.layers import GatedGraphConv
from k3_node.utils import edge_index_to_adjacency_matrix


@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.parametrize("in_channels", [32])
@pytest.mark.parametrize("out_channels", [32, 64])
@pytest.mark.parametrize("n_layers", [1, 2])
def test_gated_conv(in_channels, out_channels, n_layers, use_bias):
    x = random.normal((4, in_channels))
    edge_index = [[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]]
    adj1 = edge_index_to_adjacency_matrix(edge_index)

    conv = GatedGraphConv(out_channels, n_layers, use_bias=use_bias)
    out = conv((x, adj1))
    assert ops.shape(out) == (4, out_channels)
