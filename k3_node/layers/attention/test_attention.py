import pytest
from keras import ops, random

from k3_node.layers import PerformerAttention, PolynormerAttention, QFormer


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



@pytest.mark.parametrize("in_channels", [32, 64])
@pytest.mark.parametrize("heads", [4, 8])
@pytest.mark.parametrize("num_nodes", [4, 8])
def test_polynormer_attention(num_nodes, in_channels, heads):

    x = random.normal((1, num_nodes, in_channels))

    mask = ops.ones([1, num_nodes])

    attn = PolynormerAttention(
        channels=in_channels,
        heads=heads,
    )

    out = attn(x, mask)

    assert out.shape == (1, num_nodes, heads * 64)

@pytest.mark.parametrize("input_dim", [16, 32])
@pytest.mark.parametrize("hidden_dim", [16, 32])
@pytest.mark.parametrize("output_dim", [16, 32])
@pytest.mark.parametrize("num_heads", [2, 4])
@pytest.mark.parametrize("num_layers", [1, 2])
@pytest.mark.parametrize("num_nodes", [4, 8])
def test_qformer(
    input_dim,
    hidden_dim,
    output_dim,
    num_heads,
    num_layers,
    num_nodes,
):
    x = random.normal((1, num_nodes, input_dim))

    attn = QFormer(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_heads=num_heads,
        num_layers=num_layers,
    )

    out = attn(x)

    assert out.shape == (
        1,
        num_nodes,
        output_dim,
    )