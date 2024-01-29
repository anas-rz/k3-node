import pytest
from keras import ops, random

from k3_node.layers import PerformerAttention


@pytest.mark.parametrize("in_channels", [32, 64])
@pytest.mark.parametrize("out_channels", [32, 64])
@pytest.mark.parametrize("heads", [4, 8])
@pytest.mark.parametrize("num_nodes", [4, 8])
def test_performer_attention(num_nodes, in_channels, out_channels, heads):
    x = random.normal((1, num_nodes, in_channels))
    mask = ops.ones([1, num_nodes])
    attn = PerformerAttention(channels=out_channels, heads=heads)
    out = attn(x, mask)
    assert out.shape == (1, num_nodes, out_channels)
