import pytest
from keras import ops, random

from k3_node.layers import APPNPConv
from k3_node.utils import edge_index_to_adjacency_matrix


@pytest.mark.parametrize("channels", [32, 64])
# @pytest.mark.parametrize("propagations", [1, 2, 3, 4])
@pytest.mark.parametrize("use_bias", [True, False])
def test_appnp_conv(channels, use_bias):
    x = random.normal((4, 16))
    edge_index = [[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]]
    adj1 = edge_index_to_adjacency_matrix(edge_index)

    conv = APPNPConv(
        channels,
        use_bias=use_bias,
    )
    out = conv((x, adj1))
    assert ops.shape(out) == (4, channels)
