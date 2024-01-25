import pytest
from keras import ops, random

from k3_node.layers import CrystalConv
from k3_node.utils import edge_index_to_adjacency_matrix

@pytest.mark.parametrize('aggregate', ['sum', 'max'])
@pytest.mark.parametrize('use_bias', [True, False])
@pytest.mark.parametrize('in_channels', [32, 64])
def test_crystal_conv(in_channels, 
                    aggregate,
                    use_bias):
    x = random.normal((4, in_channels))
    edge_index = ([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    adj1 = edge_index_to_adjacency_matrix(edge_index)

    conv = CrystalConv(
        aggregate=aggregate,
        use_bias=use_bias)
    out = conv((x, adj1))
    assert ops.shape(out) == (4, in_channels)