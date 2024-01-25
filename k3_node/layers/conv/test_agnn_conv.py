import pytest
from keras import ops, random

from k3_node.layers import AGNNConv
from k3_node.utils import edge_index_to_adjacency_matrix


@pytest.mark.parametrize("trainable", [True, False])
def test_agnn_conv(trainable):
    x = random.normal((4, 16))
    edge_index = [[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]]
    adj1 = edge_index_to_adjacency_matrix(edge_index)

    conv = AGNNConv(trainable=trainable)
    out = conv(x, adj1)
    assert ops.shape(out) == (4, 16)
